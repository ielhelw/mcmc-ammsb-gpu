#include "mcmc/cuckoo.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

using namespace mcmc::cuckoo;
namespace compute = boost::compute;

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

std::string kSource = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void find(__global void *vset, __global uint64_t *in, uint32_t len,
                       __global uint64_t *out) {
      size_t id = get_global_id(0);
      for (; id < len; id += get_global_size(0)) {
        __global Set *set = (__global Set *)vset;
        out[id] = Set_HasEdge(set, in[id]) ? 1 : 0;
      }
    });

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
  compute::device dev = compute::system::default_device();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  compute::program prog =
      compute::program::create_with_source(OpenClSetFactory::GetHeader() + kSource, context);
  try {
    prog.build();
  } catch (compute::opencl_error &e) {
    LOG(FATAL) << prog.build_log();
    ;
  }
  auto set_factory = OpenClSetFactory::New(queue);
  std::unique_ptr<OpenClSet> dev_set(set_factory->CreateSet(serialized_set));
  compute::vector<Edge> dev_input(edges.begin(), edges.begin() + in_len, queue);
  compute::vector<Edge> dev_output(dev_input.size(), 0, queue);
  compute::kernel kernel = prog.create_kernel("find");
  int arg = 0;
  kernel.set_arg(arg++, dev_set->Get());
  kernel.set_arg(arg++, dev_input);
  kernel.set_arg(arg++, static_cast<uint32_t>(dev_input.size()));
  kernel.set_arg(arg++, dev_output);
  uint64_t local = 64;
  uint64_t global = static_cast<uint64_t>(ceil((in_len + 0.0) / local) * local);
  compute::event e = queue.enqueue_1d_range_kernel(kernel, 0, global, local);
  e.wait();
  std::vector<Edge> output(dev_output.size());
  compute::copy(dev_output.begin(), dev_output.end(), output.begin(), queue);
  uint64_t count = 0;
  for (auto v : output) {
    if (v != 1) ++count;
  }
  ASSERT_EQ(count, 0);
  dev_input.resize(out_len);
  dev_output.resize(dev_input.size());
  compute::copy(edges.begin() + in_len, edges.end(), dev_input.begin(), queue);
  compute::fill(dev_output.begin(), dev_output.end(), 1, queue);
  arg = 0;
  kernel.set_arg(arg++, dev_set->Get());
  kernel.set_arg(arg++, dev_input);
  kernel.set_arg(arg++, static_cast<uint32_t>(dev_input.size()));
  kernel.set_arg(arg++, dev_output);
  global = static_cast<uint64_t>(ceil((out_len + 0.0) / local) * local);
  e = queue.enqueue_1d_range_kernel(kernel, 0, global, local);
  e.wait();
  output.resize(out_len);
  compute::copy(dev_output.begin(), dev_output.end(), output.begin(), queue);
  count = 0;
  for (auto v : output) {
    if (v != 0) ++count;
  }
  ASSERT_EQ(count, 0);
}
