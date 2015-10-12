#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

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
  compute::kernel kernel = prog_.create_kernel("WG_SUM_KERNEL_uint");
  for (auto v : vals) {
    std::vector<uint32_t> host(v);
    for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
    std::random_shuffle(host.begin(), host.end());
    compute::vector<uint32_t> in(host.begin(), host.end(), queue_);
    compute::vector<uint32_t> out(1, context_);
    uint32_t scratch_per_wg =
        static_cast<size_t>(std::ceil(in.size() / static_cast<double>(wg)));
    compute::vector<uint32_t> scratch(scratch_per_wg, context_);
    kernel.set_arg(0, in);
    kernel.set_arg(1, out);
    kernel.set_arg(2, scratch);
    kernel.set_arg(3, wg * sizeof(uint32_t), 0);
    kernel.set_arg(4, static_cast<compute::uint_>(in.size()));
    auto e = queue_.enqueue_1d_range_kernel(kernel, 0, wg, wg);
    e.wait();
    compute::copy(out.begin(), out.begin() + 1, host.begin(), queue_);
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
  compute::vector<compute::uint_> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  uint32_t wg = 32;
  uint32_t scratch_per_wg =
      static_cast<uint32_t>(std::ceil(K / static_cast<double>(wg)));
  compute::vector<compute::uint_> scratch(N * scratch_per_wg, context_);
  compute::vector<compute::uint_> out(N, context_);
  auto t1 = std::chrono::high_resolution_clock::now();
  compute::kernel kernel = prog_.create_kernel("WG_SUM_KERNEL_uint");
  kernel.set_arg(0, in);
  kernel.set_arg(1, out);
  kernel.set_arg(2, scratch);
  kernel.set_arg(3, wg * sizeof(uint32_t), 0);
  kernel.set_arg(4, static_cast<compute::uint_>(K));
  auto e = queue_.enqueue_1d_range_kernel(kernel, 0, N * wg, wg);
  e.wait();
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(out.begin() + i, out.begin() + i + 1, host.begin(), queue_);
    ASSERT_EQ((K * (K + 1)) / 2, host[0]);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "custom: " << (t2 - t1).count();
}

TEST_F(ContextTest, BoostSumPerformance) {
  std::vector<uint32_t> host(K);
  for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
  compute::vector<compute::uint_> in(N * K, context_);
  for (uint32_t i = 0; i < N; ++i) {
    compute::copy(host.begin(), host.end(), in.begin() + i * K, queue_);
  }
  compute::vector<compute::uint_> out(1, context_);
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

TEST_F(WgSumTest, PartitionedTest) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  uint32_t num_blocks =
      rows / num_rows_in_block + (rows % num_rows_in_block ? 1 : 0);
  uint32_t num_elements_in_block = num_rows_in_block * cols;
  auto factory = RowPartitionedMatrixFactory<compute::uint_>::New(queue_);
  std::unique_ptr<RowPartitionedMatrix<compute::uint_>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  std::vector<compute::uint_> host(cols);
  uint32_t n = 0;
  std::generate(host.begin(), host.end(), [&n]() { return ++n; });
  for (uint32_t i = 0; i < p->Blocks().size(); ++i) {
    for (uint32_t j = 0; j < p->Blocks()[i].size() / cols; ++j) {
      compute::copy(host.begin(), host.end(), p->Blocks()[i].begin() + j * cols,
                    queue_);
    }
  }
  uint32_t wg = 32;
  uint32_t scratch_per_wg =
      static_cast<uint32_t>(std::ceil(cols / static_cast<double>(wg)));
  compute::vector<compute::uint_> scratch(rows * scratch_per_wg, context_);
  compute::vector<compute::uint_> out(rows, context_);
  compute::kernel kernel =
      prog_.create_kernel("WG_SUM_PARTITIONED_KERNEL_uint");
  kernel.set_arg(0, p->Get());
  kernel.set_arg(1, out);
  kernel.set_arg(2, scratch);
  kernel.set_arg(3, wg * sizeof(uint32_t), 0);
  auto e = queue_.enqueue_1d_range_kernel(kernel, 0, rows * wg, wg);
  e.wait();
  for (uint32_t i = 0; i < rows; ++i) {
    compute::copy(out.begin() + i, out.begin() + i + 1, host.begin(), queue_);
    ASSERT_EQ((cols * (cols + 1)) / 2, host[0]);
  }
}

}  // namespace test
}  // namespace mcmc
