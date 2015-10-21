#include "mcmc/cuckoo.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

using namespace mcmc::cuckoo;
namespace clcuda = mcmc::clcuda;

std::vector<uint64_t> GenerateRandom() {
  std::vector<Edge> edges(2 * 1024 * 1024);
  std::default_random_engine generator;
  std::uniform_int_distribution<Edge> distribution(
      0, std::numeric_limits<Edge>::max());
  for (auto &e : edges) {
    e = distribution(generator);
  }
  std::sort(edges.begin(), edges.end());
  auto end = std::unique(edges.begin(), edges.end());
  edges.resize(end - edges.begin());
  return edges;
}

TEST(CuckooSetTest, RandomMembership) {
  std::vector<Edge> edges = GenerateRandom();
  uint64_t in_len = static_cast<uint64_t>(1 + ceil(edges.size() / 2.0));
  uint64_t out_len = edges.size() - in_len;
  ASSERT_GT(in_len, 0);
  ASSERT_GT(out_len, 0);
  Set set(in_len);
  for (auto it = edges.begin(); it != edges.begin() + in_len; ++it) {
    ASSERT_TRUE(set.Insert(*it));
  }
  for (auto it = edges.begin(); it != edges.begin() + in_len; ++it) {
    ASSERT_TRUE(set.Has(*it));
  }
  for (auto it = edges.begin() + in_len; it != edges.end(); ++it) {
    ASSERT_FALSE(set.Has(*it));
  }
}

std::string kSource = R"%%(
    KERNEL void find(GLOBAL void *vset, GLOBAL uint64_t *in, uint32_t len,
                       GLOBAL uint64_t *out) {
      size_t id = GET_GLOBAL_ID();
      for (; id < len; id += GET_GLOBAL_SIZE()) {
        GLOBAL Set *set = (GLOBAL Set *)vset;
        out[id] = Set_HasEdge(set, in[id]) ? 1 : 0;
      }
    })%%";

TEST(OpenClCuckooSetTest, RandomMembership) {
  std::vector<Edge> edges = GenerateRandom();
  uint64_t in_len = static_cast<uint64_t>(1 + ceil(edges.size() / 2.0));
  uint64_t out_len = edges.size() - in_len;
  ASSERT_GT(in_len, 0);
  ASSERT_GT(out_len, 0);
  Set set(in_len);
  for (auto it = edges.begin(); it != edges.begin() + in_len; ++it) {
    ASSERT_TRUE(set.Insert(*it));
  }
  std::vector<Edge> serialized_set = set.Serialize();
  clcuda::Platform platform((size_t)0);
  clcuda::Device dev(platform, 0);
  clcuda::Context context(dev);
  clcuda::Queue queue(context, dev);
  uint64_t local = 64;
  clcuda::Program prog(
      context, OpenClSetFactory::GetHeader() + ::mcmc::GetClTypes() + kSource);
  std::vector<std::string> opts = ::mcmc::GetClFlags(local);
  clcuda::BuildStatus status = prog.Build(dev, opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog.GetBuildInfo(dev);
  auto set_factory = OpenClSetFactory::New(queue);
  std::unique_ptr<OpenClSet> dev_set(set_factory->CreateSet(serialized_set));
  std::unique_ptr<clcuda::Buffer<Edge>> dev_input(new clcuda::Buffer<Edge>(
      context, queue, edges.begin(), edges.begin() + in_len));
  std::unique_ptr<clcuda::Buffer<Edge>> dev_output(
      new clcuda::Buffer<Edge>(context, in_len));
  clcuda::Kernel kernel(prog, "find");
  int arg = 0;
  kernel.SetArgument(arg++, dev_set->Get()());
  kernel.SetArgument(arg++, *dev_input);
  kernel.SetArgument(arg++, static_cast<uint32_t>(in_len));
  kernel.SetArgument(arg++, *dev_output);
  uint64_t global = static_cast<uint64_t>(ceil((in_len + 0.0) / local) * local);
  clcuda::Event e;
  kernel.Launch(queue, {global}, {local}, e);
  queue.Finish();
  std::vector<Edge> output(in_len);
  dev_output->Read(queue, dev_output->GetSize() / sizeof(Edge), output.data());
  uint64_t count = 0;
  for (auto v : output) {
    if (v != 1) ++count;
  }
  ASSERT_EQ(count, 0);
  dev_input.reset(new clcuda::Buffer<Edge>(
      context, queue, edges.begin() + in_len, edges.end()));
  dev_output.reset(new clcuda::Buffer<Edge>(context, out_len));
  arg = 0;
  kernel.SetArgument(arg++, dev_set->Get()());
  kernel.SetArgument(arg++, *dev_input);
  kernel.SetArgument(arg++, static_cast<uint32_t>(out_len));
  kernel.SetArgument(arg++, *dev_output);
  global = static_cast<uint64_t>(ceil((out_len + 0.0) / local) * local);
  kernel.Launch(queue, {global}, {local}, e);
  queue.Finish();
  output.resize(out_len);
  dev_output->Read(queue, out_len, output.data());
  count = 0;
  for (auto v : output) {
    if (v != 0) ++count;
  }
  ASSERT_EQ(count, 0);
}
