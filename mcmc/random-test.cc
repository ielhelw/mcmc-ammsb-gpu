#include "mcmc/random.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

using namespace mcmc::random;
namespace compute = boost::compute;
using mcmc::Float;

const std::string kSource =
    ::mcmc::random::GetRandomHeader() +

    BOOST_COMPUTE_STRINGIZE_SOURCE(
        KERNEL void test(GLOBAL void* vrand, GLOBAL ulong* ok) {
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
        }

        KERNEL void generate(GLOBAL void* vrand,
                               GLOBAL Float* data,  // [#threads * K]
                               uint K) {
          uint gid = GET_GLOBAL_ID();
          data += gid * K;
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          random_seed_t seed = rand->base_[gid];
          for (uint i = 0; i < K; ++i) {
            data[i] = randn(&seed);
          }
          rand->base_[gid] = seed;
        });

TEST(RandomTest, Check) {
  compute::device dev = compute::system::default_device();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  random_seed_t seed;
  seed[0] = 42;
  seed[1] = 43;
  std::vector<random_seed_t> host(1000);
  auto factory = OpenClRandomFactory::New(queue);
  std::unique_ptr<OpenClRandom> random(
      factory->CreateRandom(host.size(), seed));
  compute::copy(random->GetSeeds().begin(), random->GetSeeds().end(),
                host.begin(), queue);
  for (size_t i = 0; i < host.size(); ++i) {
    ASSERT_EQ(seed[0] + i, host[i][0]);
    ASSERT_EQ(seed[1] + i, host[i][1]);
  }

  compute::program prog =
      compute::program::create_with_source(kSource, context);
  try {
    prog.build();
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog.build_log();
  }
  compute::vector<uint64_t> dev_ok(1, static_cast<uint64_t>(0), queue);
  compute::kernel kernel = prog.create_kernel("test");
  kernel.set_arg(0, random->Get());
  kernel.set_arg(1, dev_ok);
  compute::event e = queue.enqueue_task(kernel);
  e.wait();
  std::vector<uint64_t> ok(dev_ok.size(), 0);
  compute::copy(dev_ok.begin(), dev_ok.end(), ok.begin(), queue);
  ASSERT_EQ(1, ok[0]);
  uint32_t K = 10000;
  compute::vector<Float> data(host.size() * K, context);
  compute::kernel generate = prog.create_kernel("generate");
  generate.set_arg(0, random->Get());
  generate.set_arg(1, data);
  generate.set_arg(2, K);
  e = queue.enqueue_1d_range_kernel(generate, 0, host.size(), 0);
  e.wait();
  std::vector<Float> hdata(data.size());
  compute::copy(data.begin(), data.end(), hdata.begin(), queue);
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
