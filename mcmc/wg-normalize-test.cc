#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/normalize.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class WgNormalizeTest : public ContextTest {
 protected:
  WgNormalizeTest()
      : ContextTest(mcmc::algorithm::WorkGroupNormalize("float")) {}
};

class WgNormalizeParameterizedTest
    : public WgNormalizeTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(WgNormalizeParameterizedTest, VaryLength) {
  uint32_t wg = GetParam();
  std::vector<uint32_t> vals = {1,  2,   3,   4,    5,    6,    7,  11,
                                31, 32,  33,  47,   48,   49,   63, 64,
                                65, 127, 128, 1023, 1024, 11331};
  compute::kernel kernel = prog_.create_kernel("WG_NORMALIZE_KERNEL_float");
  for (auto v : vals) {
    std::vector<compute::float_> host(v);
    for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
    compute::vector<compute::float_> in(host.begin(), host.end(), queue_);
    compute::vector<compute::float_> scratch(
        static_cast<size_t>(ceil(in.size() / static_cast<float>(wg))),
        context_);
    kernel.set_arg(0, in);
    kernel.set_arg(1, scratch);
    kernel.set_arg(2, wg * sizeof(compute::float_), 0);
    kernel.set_arg(3, static_cast<compute::uint_>(in.size()));
    auto e = queue_.enqueue_1d_range_kernel(kernel, 0, wg, wg);
    e.wait();
    compute::copy(in.begin(), in.end(), host.begin(), queue_);
    float sum = (v * (v + 1)) / 2.0;
    for (uint32_t i = 0; i < host.size(); ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
}
INSTANTIATE_TEST_CASE_P(WorkGroups, WgNormalizeParameterizedTest,
                        ::testing::ValuesIn(std::vector<uint32_t>({2, 4, 16, 32,
                                                                   64})));

const uint32_t N = 64;
const uint32_t K = 1024;

TEST_F(WgNormalizeTest, CustomNormalizePerformance) {
  std::vector<float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<compute::float_> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  uint32_t wg = 32;
  uint32_t scratch_per_wg =
      static_cast<uint32_t>(std::ceil(K / static_cast<float>(wg)));
  compute::vector<compute::float_> scratch(N * scratch_per_wg, context_);
  auto t1 = std::chrono::high_resolution_clock::now();
  compute::kernel kernel = prog_.create_kernel("WG_NORMALIZE_KERNEL_float");
  kernel.set_arg(0, in);
  kernel.set_arg(1, scratch);
  kernel.set_arg(2, wg * sizeof(compute::float_), 0);
  kernel.set_arg(3, static_cast<compute::uint_>(K));
  auto e = queue_.enqueue_1d_range_kernel(kernel, 0, N * wg, wg);
  e.wait();
  float sum = (K * (K + 1)) / 2.0;
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(scratch.begin() + i * scratch_per_wg,
                  scratch.begin() + i * scratch_per_wg + 1, host.begin(),
                  queue_);
    ASSERT_FLOAT_EQ(sum, host[0]);
    compute::copy(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                  queue_);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "custom: " << (t2 - t1).count();
}

TEST_F(ContextTest, BoostNormalizePerformance) {
  std::vector<float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<compute::float_> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  float sum = (K * (K + 1)) / 2.0;
  for (uint32_t i = 0; i < N; ++i) {
    compute::reduce(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                    queue_);
    ASSERT_FLOAT_EQ(sum, host[0]);
    compute::transform(in.begin() + i * K, in.begin() + (i + 1) * K,
                       in.begin() + i * K, compute::_1 / sum, queue_);
    compute::copy(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                  queue_);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "compute: " << (t2 - t1).count();
}

}  // namespace test
}  // namespace mcmc
