#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>

#include "mcmc/algorithm/sum.h"
#include "mcmc/partitioned-alloc.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class WgSumTest : public ContextTest {
 protected:
  WgSumTest() : ContextTest(mcmc::algorithm::WorkGroupSum("uint")) {}
};

class WgSumParameterizedTest : public WgSumTest,
                               public ::testing::WithParamInterface<uint32_t> {
};

TEST_P(WgSumParameterizedTest, VaryLength) {
  uint32_t wg = GetParam();
  std::vector<uint32_t> vals = {1,  2,   3,   4,    5,    6,    7,  11,
                                31, 32,  33,  47,   48,   49,   63, 64,
                                65, 127, 128, 1023, 1024, 11331};
  clcuda::Kernel kernel(*prog_, "WG_SUM_KERNEL_uint");
  for (auto v : vals) {
    std::vector<uint32_t> host(v);
    for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
    std::random_shuffle(host.begin(), host.end());
    clcuda::Buffer<uint32_t> in(*context_, *queue_, host.begin(), host.end());
    clcuda::Buffer<uint32_t> out(*context_, 1);
    kernel.SetArgument(0, in);
    kernel.SetArgument(1, out);
    kernel.SetArgument(2, static_cast<uint32_t>(host.size()));
    clcuda::Event e;
    kernel.Launch(*queue_, {wg}, {wg}, e);
    queue_->Finish();
    out.Read(*queue_, 1, host);
    ASSERT_EQ((v * (v + 1)) / 2, host[0]);
  }
}
INSTANTIATE_TEST_CASE_P(WorkGroups, WgSumParameterizedTest,
                        ::testing::ValuesIn(std::vector<uint32_t>({2, 4, 16, 32,
                                                                   64})));

const uint32_t N = 1024;
const uint32_t K = 1024;

TEST_F(WgSumTest, CustomSumPerformance) {
  std::vector<uint32_t> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  clcuda::Buffer<uint32_t> in(*context_, N * K);
  for (uint32_t i = 0; i < N; ++i) {
    in.Write(*queue_, host.size(), host, i * K);
  }
  uint32_t wg = 32;
  clcuda::Buffer<uint32_t> out(*context_, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  clcuda::Kernel kernel(*prog_, "WG_SUM_KERNEL_uint");
  kernel.SetArgument(0, in);
  kernel.SetArgument(1, out);
  kernel.SetArgument(2, static_cast<uint32_t>(K));
  clcuda::Event e;
  kernel.Launch(*queue_, {N * wg}, {wg}, e);
  queue_->Finish();
  for (uint32_t i = 0; i < N; ++i) {
    out.Read(*queue_, 1, host, i);
    ASSERT_EQ((K * (K + 1)) / 2, host[0]);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "custom: " << (t2 - t1).count();
}

#if 0
TEST_F(ContextTest, BoostSumPerformance) {
  std::vector<uint32_t> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  clcuda::Buffer<uint32_t> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  clcuda::Buffer<uint32_t> out(1, context_);
  auto t1 = std::chrono::high_resolution_clock::now();
  for (uint32_t i = 0; i < N; ++i) {
    compute::reduce(in.begin() + i * K, in.begin() + (i + 1) * K, out.begin(),
                    queue_);
    compute::copy(out.begin(), out.begin() + 1, host.begin(), queue_);
    ASSERT_EQ((K * (K + 1)) / 2, host[0]);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "compute: " << (t2 - t1).count();
}
#endif

TEST_F(WgSumTest, PartitionedTest) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  auto factory = RowPartitionedMatrixFactory<uint32_t>::New(*queue_);
  std::unique_ptr<RowPartitionedMatrix<uint32_t>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  std::vector<uint32_t> host(cols);
  uint32_t n = 0;
  std::generate(host.begin(), host.end(), [&n]() { return ++n; });
  for (uint32_t i = 0; i < p->Blocks().size(); ++i) {
    for (uint32_t j = 0; j < (p->Blocks()[i].GetSize()/sizeof(uint32_t)) / cols; ++j) {
      p->Blocks()[i].Write(*queue_, host.size(), host, j * cols);
    }
  }
  uint32_t wg = 32;
  clcuda::Buffer<uint32_t> out(*context_, rows);
  clcuda::Kernel kernel(*prog_, "WG_SUM_PARTITIONED_KERNEL_uint");
  kernel.SetArgument(0, p->Get());
  kernel.SetArgument(1, out);
  clcuda::Event e;
  kernel.Launch(*queue_, {rows * wg}, {wg}, e);
  queue_->Finish();
  for (uint32_t i = 0; i < rows; ++i) {
    out.Read(*queue_, 1, host, i);
    ASSERT_EQ((cols * (cols + 1)) / 2, host[0]);
  }
}

}  // namespace test
}  // namespace mcmc
