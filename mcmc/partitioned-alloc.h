#ifndef __MCMC_PARTITIONED_ALLOC_H__
#define __MCMC_PARTITIONED_ALLOC_H__

#include <boost/compute/container/vector.hpp>
#include <boost/compute/utility/source.hpp>
#include <glog/logging.h>

#include "mcmc/gen-util.h"
#include "mcmc/types.h"

namespace mcmc {

inline std::string GetRowPartitionedMatrixSource() {
  static const std::string kSourceRowPartitionedMatrix =
      BOOST_COMPUTE_STRINGIZE_SOURCE(

          typedef struct {
            __global TT* blocks_[32];
            uint num_rows_in_block_;
            uint num_blocks_;
            uint num_rows_;
            uint num_cols_;
          } TTRowPartitionedMatrix;

          __global TT * TTRowPartitionedMatrix_Row(
                            __global TTRowPartitionedMatrix * pm, uint row) {
                          uint blockIdx = row / pm->num_rows_in_block_;
                          __global TT* block = pm->blocks_[blockIdx];
                          uint rowOffsetInBlock =
                              (row % pm->num_rows_in_block_) * pm->num_cols_;
                          return block + rowOffsetInBlock;
                        }

              __kernel void TTRowPartitionedMatrix_init(
                  __global void* vpm, uint num_rows_in_block, uint num_blocks,
                  uint num_rows, uint num_cols) {
                __global TTRowPartitionedMatrix* pm =
                    (__global TTRowPartitionedMatrix*)vpm;
                pm->num_rows_in_block_ = num_rows_in_block;
                pm->num_blocks_ = num_blocks;
                pm->num_rows_ = num_rows;
                pm->num_cols_ = num_cols;
              }

              __kernel void TTRowPartitionedMatrix_set(
                  __global void* vpm, __global void* block, uint idx) {
                __global TTRowPartitionedMatrix* pm =
                    (__global TTRowPartitionedMatrix*)vpm;
                pm->blocks_[idx] = (__global TT*)block;
              }

              __kernel void TTRowPartitionedMatrixSizeof(__global ulong *
                                                         size) {
            size[0] = sizeof(TTRowPartitionedMatrix);
          }

          );
  return kSourceRowPartitionedMatrix;
}

inline std::string GetRowPartitionedMatrixHeader(const std::string& type) {
  return gen::MakeHeaderFromTemplate(type + "RowPartitionedMatrix",
                                     GetRowPartitionedMatrixSource(), "TT",
                                     type);
}

template <class T>
std::string GetRowPartitionedMatrixHeader() {
  return gen::MakeHeaderFromTemplate(
      std::string(compute::type_name<T>()) + "RowPartitionedMatrix",
      GetRowPartitionedMatrixSource(), "TT", compute::type_name<T>());
}

template <class T>
class RowPartitionedMatrixFactory;

template <class T>
class RowPartitionedMatrix {
 public:
  uint32_t Rows() const { return rows_; }

  uint32_t Cols() const { return cols_; }

  uint32_t RowsPerBlock() const { return rows_per_alloc_; }

  std::vector<compute::vector<T>>& Blocks() { return blocks_; }

  compute::buffer& Get() { return base_; }

 private:
  RowPartitionedMatrix(compute::command_queue queue, uint32_t rows,
                       uint32_t cols, uint32_t sizeOf, compute::kernel init,
                       compute::kernel set, uint32_t rowsInBlock = 0)
      : queue_(queue),
        rows_(rows),
        cols_(cols),
        rows_per_alloc_(rowsInBlock == 0
                            ? GetMaxRowsInBlock(queue.get_device(), cols_)
                            : rowsInBlock),
        base_(queue_.get_context(), sizeOf) {
    for (uint32_t i = 0; i < rows_ / rows_per_alloc_; ++i) {
      blocks_.push_back(
          compute::vector<T>(rows_per_alloc_ * cols_, queue_.get_context()));
    }
    if (rows_ % rows_per_alloc_) {
      blocks_.push_back(compute::vector<T>((rows_ % rows_per_alloc_) * cols_,
                                           queue_.get_context()));
    }
    init.set_arg(0, base_);
    init.set_arg(1, rows_per_alloc_);
    init.set_arg(2, static_cast<uint32_t>(blocks_.size()));
    init.set_arg(3, rows_);
    init.set_arg(4, cols_);
    auto e = queue_.enqueue_task(init);
    e.wait();
    set.set_arg(0, base_);
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
      set.set_arg(1, blocks_[i]);
      set.set_arg(2, i);
      e = queue_.enqueue_task(set);
      e.wait();
    }
  }

  uint64_t GetMaxRowsInBlock(compute::device dev, uint32_t cols) {
    uint64_t row_size = cols * sizeof(T);
    compute::ulong_ max_alloc =
        dev.get_info<compute::ulong_>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    LOG(INFO) << "MAX ALLOC = " << max_alloc;
    return max_alloc / row_size;
  }

  compute::command_queue queue_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t rows_per_alloc_;
  std::vector<compute::vector<T>> blocks_;
  compute::buffer base_;

  friend class RowPartitionedMatrixFactory<T>;
};

template <class T>
class RowPartitionedMatrixFactory
    : public std::enable_shared_from_this<RowPartitionedMatrixFactory<T>> {
 public:
  static std::shared_ptr<RowPartitionedMatrixFactory> New(
      compute::command_queue queue) {
    return std::shared_ptr<RowPartitionedMatrixFactory>(
        new RowPartitionedMatrixFactory(queue));
  }

  RowPartitionedMatrix<T>* CreateMatrix(uint32_t rows, uint32_t cols,
                                        uint32_t rowsInBlock = 0) {
    RowPartitionedMatrix<T>* rpm = new RowPartitionedMatrix<T>(
        queue_, rows, cols, sizeOf_, init_kernel_, set_kernel_, rowsInBlock);
    return rpm;
  }

 private:
  RowPartitionedMatrixFactory(compute::command_queue queue) : queue_(queue) {
    prog_ = compute::program::create_with_source(
        GetRowPartitionedMatrixHeader<T>(), queue_.get_context());
    try {
      prog_.build();
    } catch (compute::opencl_error& e) {
      LOG(FATAL) << prog_.build_log();
    }
    compute::vector<compute::ulong_> sizeOf(1, queue_.get_context());
    compute::kernel sizeofKernel = prog_.create_kernel(
        std::string(compute::type_name<T>()) + "RowPartitionedMatrixSizeof");
    sizeofKernel.set_arg(0, sizeOf);
    auto e = queue_.enqueue_task(sizeofKernel);
    e.wait();
    compute::copy(sizeOf.begin(), sizeOf.begin() + 1, &sizeOf_, queue_);
    init_kernel_ = prog_.create_kernel(std::string(compute::type_name<T>()) +
                                       "RowPartitionedMatrix_init");
    set_kernel_ = prog_.create_kernel(std::string(compute::type_name<T>()) +
                                      "RowPartitionedMatrix_set");
  }

  compute::command_queue queue_;
  compute::program prog_;
  compute::kernel init_kernel_;
  compute::kernel set_kernel_;
  compute::ulong_ sizeOf_;
};

}  // namespace mcmc

#endif  // __MCMC_PARTITIONED_ALLOC_H__
