#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>

#include "mcmc/test.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {
namespace test {

class PartitionedAllocTest : public ContextTest {};

TEST_F(PartitionedAllocTest, CountBlocksAndElementsTest) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  uint32_t num_blocks =
      rows / num_rows_in_block + (rows % num_rows_in_block ? 1 : 0);
  uint32_t num_elements_in_block = num_rows_in_block * cols;

  auto factory = RowPartitionedMatrixFactory<uint32_t>::New(*queue_);
  std::unique_ptr<RowPartitionedMatrix<uint32_t>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  ASSERT_EQ(num_blocks, p->Blocks().size());
  uint32_t i = 0;
  for (; i < num_blocks - 1; ++i) {
    ASSERT_LT(i, p->Blocks().size());
    ASSERT_EQ(num_elements_in_block,
              p->Blocks()[i].GetSize() / sizeof(uint32_t));
  }
  if (num_blocks > i) {
    ASSERT_LT(i, p->Blocks().size());
    ASSERT_EQ(rows * cols - i * num_elements_in_block,
              p->Blocks()[i].GetSize() / sizeof(uint32_t));
  }
}

TEST_F(PartitionedAllocTest, RetrieveElements) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  uint32_t num_blocks =
      rows / num_rows_in_block + (rows % num_rows_in_block ? 1 : 0);
  uint32_t num_elements_in_block = num_rows_in_block * cols;
  auto factory = RowPartitionedMatrixFactory<uint32_t>::New(*queue_);
  std::unique_ptr<RowPartitionedMatrix<uint32_t>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  ASSERT_EQ(num_blocks, p->Blocks().size());

  const std::string kSource = R"%%(

      KERNEL void fetch(GLOBAL void* g_matrix, uint row, uint col,
                        GLOBAL uint* ret) {
        GLOBAL uintRowPartitionedMatrix* rpm =
            (GLOBAL uintRowPartitionedMatrix*)g_matrix;
        GLOBAL uint* p = uintRowPartitionedMatrix_Row(rpm, row);
        ret[0] = p[col];
        ret[1] = p[col + 1];
      }

      )%%";

  clcuda::Program prog(*context_,
                       GetRowPartitionedMatrixHeader<uint32_t>() + kSource);
  std::vector<std::string> opts = GetClFlags();
  LOG_IF(FATAL, prog.Build(*device_, opts) != clcuda::BuildStatus::kSuccess)
      << prog.GetBuildInfo(*device_);
  std::vector<uint32_t> host(2);
  std::vector<uint32_t> hret(2);
  clcuda::Buffer<uint32_t> dev_ret(*context_, 2);
  clcuda::Kernel fetch(prog, "fetch");
  fetch.SetArgument(0, p->Get());
  fetch.SetArgument(3, dev_ret);

  for (uint32_t i = 0; i < 100; ++i) {
    host[0] = rand();
    host[1] = rand();
    // get random row/col
    uint32_t r = rand() % rows;
    uint32_t c = rand() % (cols - 3);
    fetch.SetArgument(1, r);
    fetch.SetArgument(2, c);
    // copy data there
    p->Blocks()[r / num_rows_in_block]
        .Write(*queue_, host.size(), host, (r % num_rows_in_block) * cols + c);
    clcuda::Event e;
    fetch.Launch(*queue_, {1}, {1}, e);
    queue_->Finish();
    dev_ret.Read(*queue_, hret.size(), hret);
    ASSERT_EQ(host[0], hret[0]);
    ASSERT_EQ(host[1], hret[1]);
  }
}

}  // namespace test
}  // namespace mcmc
