#include "mcmc/random.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

using namespace mcmc::random;
using mcmc::Float;
namespace clcuda = mcmc::clcuda;

const std::string kSource = ::mcmc::random::GetRandomHeader() + R"%%(
        KERNEL void test(GLOBAL void * vrand, GLOBAL ulong * ok) {
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (rand == 0) {
            *ok = 0;
            return;
          }
          if (rand->num_seeds != 1000) {
            *ok = 0;
            return;
          }
          for (uint i = 0; i < rand->num_seeds; ++i) {
            if (rand->base_[i].x != 42 + i || rand->base_[i].y != 43 + i) {
              *ok = 0;
              return;
            }
          }
          *ok = 1;
        } KERNEL void generate(GLOBAL void * vrand,
                               GLOBAL Float * data,  // [#threads * K]
                               uint K) {
          uint gid = GET_GLOBAL_ID();
          data += gid * K;
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          random_seed_t seed = rand->base_[gid];
          for (uint i = 0; i < K; ++i) {
            data[i] = randn(&seed);
          }
          rand->base_[gid] = seed;
        })%%";

TEST(RandomTest, Check) {
  clcuda::Platform platform((size_t)0);
  clcuda::Device dev(platform, 0);
  clcuda::Context context(dev);
  clcuda::Queue queue(context, dev);
  random_seed_t seed;
  seed[0] = 42;
  seed[1] = 43;
  std::vector<random_seed_t> host(1000);
  auto factory = OpenClRandomFactory::New(queue);
  std::unique_ptr<OpenClRandom> random(
      factory->CreateRandom(host.size(), seed));
  random->GetSeeds().Read(queue, host.size(), host.data());
  for (size_t i = 0; i < host.size(); ++i) {
    ASSERT_EQ(seed[0] + i, host[i][0]);
    ASSERT_EQ(seed[1] + i, host[i][1]);
  }

  clcuda::Program prog(context, kSource);
  std::vector<std::string> opts = ::mcmc::GetClFlags();
  clcuda::BuildStatus status = prog.Build(dev, opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog.GetBuildInfo(dev);
  clcuda::Buffer<uint64_t> dev_ok(context, 1);
  clcuda::Kernel kernel(prog, "test");
  kernel.SetArgument(0, random->Get());
  kernel.SetArgument(1, dev_ok);
  clcuda::Event e;
  kernel.Launch(queue, {1}, {1}, e);
  queue.Finish();
  std::vector<uint64_t> ok(1, 0);
  dev_ok.Read(queue, 1, ok.data());
  ASSERT_EQ(static_cast<uint64_t>(1), ok[0]);
  uint32_t K = 10000;
  clcuda::Buffer<Float> data(context, host.size() * K);
  clcuda::Kernel generate(prog, "generate");
  generate.SetArgument(0, random->Get());
  generate.SetArgument(1, data);
  generate.SetArgument(2, K);
  generate.Launch(queue, {host.size()}, {1}, e);
  queue.Finish();
  std::vector<Float> hdata(data.GetSize() / sizeof(Float));
  data.Read(queue, hdata.size(), hdata.data());
  Float sum = 0;
  for (auto v : hdata) sum += v;
  Float mean = sum / hdata.size();
  Float sum_square_diff = 0;
  for (auto v : hdata) {
    sum_square_diff += std::pow(v - mean, 2);
  }
  Float stdev = std::sqrt(sum_square_diff / hdata.size());
  LOG(INFO) << "MEAN = " << mean << ", STDEV = " << stdev;
}
