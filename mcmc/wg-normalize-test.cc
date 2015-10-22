#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>

#include "mcmc/algorithm/normalize.h"
#include "mcmc/partitioned-alloc.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class WgNormalizeTest : public ContextTest {
 protected:
  WgNormalizeTest()
      : ContextTest(
            mcmc::algorithm::WorkGroupNormalizeProgram(type_name<Float>())) {}
};

class WgNormalizeParameterizedTest
    : public WgNormalizeTest,
      public ::testing::WithParamInterface<uint32_t> {};

TEST_P(WgNormalizeParameterizedTest, VaryLength) {
  uint32_t wg = GetParam();
  BuildProgram(wg);
  clcuda::Kernel kernel(
      *prog_, std::string("WG_NORMALIZE_KERNEL_") + type_name<Float>());
  std::vector<uint32_t> vals = {1,  2,   3,   4,    5,    6,    7,  11,
                                31, 32,  33,  47,   48,   49,   63, 64,
                                65, 127, 128, 1023, 1024, 11331};
  for (auto v : vals) {
    std::vector<Float> host(v);
    for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
    clcuda::Buffer<Float> in(*context_, *queue_, host.begin(), host.end());
    kernel.SetArgument(0, in);
    kernel.SetArgument(1, static_cast<uint32_t>(host.size()));
    clcuda::Event e;
    kernel.Launch(*queue_, {wg}, {wg}, e);
    queue_->Finish();
    in.Read(*queue_, host.size(), host);
    Float sum = (v * (v + 1)) / 2.0;
    for (uint32_t i = 0; i < host.size(); ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
}
INSTANTIATE_TEST_CASE_P(WorkGroups, WgNormalizeParameterizedTest,
                        ::testing::ValuesIn(std::vector<uint32_t>({2,  4, 16,
                                                                   32, 64})));

const uint32_t N = 64;
const uint32_t K = 1024;

TEST_F(WgNormalizeTest, NormalizerClassPerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  clcuda::Buffer<Float> in(*context_, N * K);
  for (uint32_t i = 0; i < N; ++i) {
    in.Write(*queue_, host.size(), host, i * K);
  }
  uint32_t wg = 32;
  mcmc::algorithm::Normalizer<Float> normalizer(*queue_, &in, K, wg);
  auto t1 = std::chrono::high_resolution_clock::now();
  normalizer();
  auto t2 = std::chrono::high_resolution_clock::now();
  Float sum = (K * (K + 1)) / 2.0;
  for (uint32_t i = 0; i < N; ++i) {
    in.Read(*queue_, K, host, i * K);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  LOG(INFO) << "custom class: " << (t2 - t1).count();
}

TEST_F(WgNormalizeTest, CustomNormalizePerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  clcuda::Buffer<Float> in(*context_, N * K);
  for (uint32_t i = 0; i < N; ++i) {
    in.Write(*queue_, host.size(), host, i * K);
  }
  uint32_t wg = 32;
  BuildProgram(wg);
  clcuda::Kernel kernel(
      *prog_, std::string("WG_NORMALIZE_KERNEL_") + type_name<Float>());
  kernel.SetArgument(0, in);
  kernel.SetArgument(1, static_cast<uint32_t>(K));
  clcuda::Event e;
  auto t1 = std::chrono::high_resolution_clock::now();
  kernel.Launch(*queue_, {N * wg}, {wg}, e);
  queue_->Finish();
  auto t2 = std::chrono::high_resolution_clock::now();
  Float sum = (K * (K + 1)) / 2.0;
  for (uint32_t i = 0; i < N; ++i) {
    in.Read(*queue_, K, host, i * K);
    for (uint32_t i = 0; i < K; ++i) {
      ASSERT_FLOAT_EQ((i + 1) / sum, host[i]);
    }
  }
  LOG(INFO) << "custom: " << (t2 - t1).count();
}

#if 0
TEST_F(ContextTest, BoostNormalizePerformance) {
  std::vector<Float> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  clcuda::Buffer<Float> in(*context_, N * K);
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
#endif

TEST_F(WgNormalizeTest, PartitionedNormalizerClassTest) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  auto factory = RowPartitionedMatrixFactory<Float>::New(*queue_);
  std::unique_ptr<RowPartitionedMatrix<Float>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  std::vector<Float> host(cols);
  uint32_t n = 0;
  std::generate(host.begin(), host.end(), [&n]() { return ++n; });
  for (uint32_t i = 0; i < p->Blocks().size(); ++i) {
    for (uint32_t j = 0; j < (p->Blocks()[i].GetSize() / sizeof(Float)) / cols;
         ++j) {
      p->Blocks()[i].Write(*queue_, host.size(), host, j * cols);
    }
  }
  clcuda::Buffer<Float> g_sum(*context_, rows);
  algorithm::PartitionedNormalizer<Float> normalizer(*queue_, p.get(), g_sum,
                                                     32);
  normalizer();
  std::vector<Float> host_sum(rows);
  g_sum.Read(*queue_, rows, host_sum);
  Float sum = (cols * (cols + 1)) / 2;
  for (uint32_t i = 0; i < p->Blocks().size(); ++i) {
    ASSERT_FLOAT_EQ(sum, host_sum[i]);
    for (uint32_t j = 0; j < (p->Blocks()[i].GetSize() / sizeof(Float)) / cols;
         ++j) {
      p->Blocks()[i].Read(*queue_, cols, host, j * cols);
      for (uint32_t k = 0; k < cols; ++k) {
        ASSERT_FLOAT_EQ((k + 1) / static_cast<Float>(sum), host[k]);
      }
    }
  }
}

}  // namespace test
}  // namespace mcmc
