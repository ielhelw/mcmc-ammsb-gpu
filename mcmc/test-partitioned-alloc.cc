#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>
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

  auto factory = RowPartitionedMatrixFactory<compute::uint_>::New(queue_);
  std::unique_ptr<RowPartitionedMatrix<compute::uint_>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));
  ASSERT_EQ(num_blocks, p->Blocks().size());
  uint32_t i = 0;
  for (; i < num_blocks - 1; ++i) {
    ASSERT_LT(i, p->Blocks().size());
    ASSERT_EQ(num_elements_in_block, p->Blocks()[i].size());
  }
  if (num_blocks > i) {
    ASSERT_LT(i, p->Blocks().size());
    ASSERT_EQ(rows * cols - i * num_elements_in_block, p->Blocks()[i].size());
  }
}

TEST_F(PartitionedAllocTest, RetrieveElements) {
  uint32_t cols = 1e3;
  uint32_t rows = 1e3;
  uint32_t num_rows_in_block = rows / 11;
  uint32_t num_blocks =
      rows / num_rows_in_block + (rows % num_rows_in_block ? 1 : 0);
  uint32_t num_elements_in_block = num_rows_in_block * cols;
  auto factory = RowPartitionedMatrixFactory<compute::uint_>::New(queue_);
  std::unique_ptr<RowPartitionedMatrix<compute::uint_>> p(
      factory->CreateMatrix(rows, cols, num_rows_in_block));

  const std::string kSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

      KERNEL void fetch(GLOBAL void* g_matrix, uint row, uint col,
                          GLOBAL uint* ret) {
        GLOBAL uintRowPartitionedMatrix* rpm =
            (GLOBAL uintRowPartitionedMatrix*)g_matrix;
        GLOBAL uint* p = uintRowPartitionedMatrix_Row(rpm, row);
        ret[0] = p[col];
        ret[1] = p[col + 1];
      }

      );

  compute::program prog;
  try {
    prog = compute::program::build_with_source(
        GetRowPartitionedMatrixHeader<compute::uint_>() + kSource, context_);
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog.build_log();
  }
  compute::vector<compute::uint_> dev_ret(2, context_);
  compute::kernel fetch = prog.create_kernel("fetch");
  fetch.set_arg(0, p->Get());
  fetch.set_arg(3, dev_ret);

  for (uint32_t i = 0; i < 100; ++i) {
    std::vector<compute::uint_> host(2);
    host[0] = rand();
    host[1] = rand();
    // get random row/col
    uint32_t r = rand() % rows;
    uint32_t c = rand() % (cols - 3);
    fetch.set_arg(1, r);
    fetch.set_arg(2, c);
    // copy data there
    compute::copy(host.begin(), host.end(),
                  p->Blocks()[r / num_rows_in_block].begin() +
                      (r % num_rows_in_block) * cols + c,
                  queue_);
    queue_.enqueue_task(fetch).wait();
    ASSERT_EQ(host[0], dev_ret[0]);
    ASSERT_EQ(host[1], dev_ret[1]);
  }
}

}  // namespace test
}  // namespace mcmc
