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
      : ContextTest(mcmc::algorithm::WorkGroupNormalizeProgram(compute::type_name<Float>())) {}
};

class WgNormalizeParameterizedTest
    : public WgNormalizeTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(WgNormalizeParameterizedTest, VaryLength) {
  uint32_t wg = GetParam();
  std::vector<uint32_t> vals = {1,  2,   3,   4,    5,    6,    7,  11,
                                31, 32,  33,  47,   48,   49,   63, 64,
                                65, 127, 128, 1023, 1024, 11331};
  compute::kernel kernel = prog_.create_kernel(std::string("WG_NORMALIZE_KERNEL_") + compute::type_name<Float>());
  for (auto v : vals) {
    std::vector<Float> host(v);
    for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
    compute::vector<Float> in(host.begin(), host.end(), queue_);
    compute::vector<Float> scratch(
        static_cast<size_t>(ceil(in.size() / static_cast<Float>(wg))),
        context_);
    kernel.set_arg(0, in);
    kernel.set_arg(1, scratch);
    kernel.set_arg(2, wg * sizeof(Float), 0);
    kernel.set_arg(3, static_cast<compute::uint_>(in.size()));
    auto e = queue_.enqueue_1d_range_kernel(kernel, 0, wg, wg);
    e.wait();
    compute::copy(in.begin(), in.end(), host.begin(), queue_);
    Float sum = (v * (v + 1)) / 2.0;
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

TEST_F(WgNormalizeTest, NormalizerClassPerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<Float> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  uint32_t wg = 32;
  mcmc::algorithm::Normalizer<Float> normalizer(queue_, &in, K, wg);
  auto t1 = std::chrono::high_resolution_clock::now();
  normalizer();
  auto t2 = std::chrono::high_resolution_clock::now();
  Float sum = (K * (K + 1)) / 2.0;
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                  queue_);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  LOG(INFO) << "custom class: " << (t2 - t1).count();
}

TEST_F(WgNormalizeTest, CustomNormalizePerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<Float> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  uint32_t wg = 32;
  uint32_t scratch_per_wg =
      static_cast<uint32_t>(std::ceil(K / static_cast<Float>(wg)));
  compute::vector<Float> scratch(N * scratch_per_wg, context_);
  compute::kernel kernel = prog_.create_kernel(std::string("WG_NORMALIZE_KERNEL_") + compute::type_name<Float>());
  kernel.set_arg(0, in);
  kernel.set_arg(1, scratch);
  kernel.set_arg(2, wg * sizeof(Float), 0);
  kernel.set_arg(3, static_cast<compute::uint_>(K));
  auto t1 = std::chrono::high_resolution_clock::now();
  auto e = queue_.enqueue_1d_range_kernel(kernel, 0, N * wg, wg);
  e.wait();
  auto t2 = std::chrono::high_resolution_clock::now();
  Float sum = (K * (K + 1)) / 2.0;
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
  LOG(INFO) << "custom: " << (t2 - t1).count();
}

TEST_F(ContextTest, BoostNormalizePerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<Float> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  Float sum = (K * (K + 1)) / 2.0;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < N; ++i) {
    compute::reduce(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                    queue_);
    compute::transform(in.begin() + i * K, in.begin() + (i + 1) * K,
                       in.begin() + i * K, compute::_1 / sum, queue_);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(in.begin() + i * K, in.begin() + (i + 1) * K, host.begin(),
                  queue_);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  LOG(INFO) << "compute: " << (t2 - t1).count();
}

}  // namespace test
}  // namespace mcmc
