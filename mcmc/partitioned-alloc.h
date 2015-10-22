#ifndef __MCMC_PARTITIONED_ALLOC_H__
#define __MCMC_PARTITIONED_ALLOC_H__

#include <glog/logging.h>

#include "mcmc/gen-util.h"
#include "mcmc/types.h"

namespace mcmc {

inline std::string GetRowPartitionedMatrixSource() {
  static const std::string kSourceRowPartitionedMatrix = R"%%(

          typedef struct {
            GLOBAL TT* blocks_[32];
            uint num_rows_in_block_;
            uint num_blocks_;
            uint num_rows_;
            uint num_cols_;
          } TTRowPartitionedMatrix;

          GLOBAL TT * TTRowPartitionedMatrix_Row(
                          GLOBAL TTRowPartitionedMatrix * pm, uint row) {
                        uint blockIdx = row / pm->num_rows_in_block_;
                        GLOBAL TT* block = pm->blocks_[blockIdx];
                        uint rowOffsetInBlock =
                            (row % pm->num_rows_in_block_) * pm->num_cols_;
                        return block + rowOffsetInBlock;
                      }

              KERNEL void TTRowPartitionedMatrix_init(
                  GLOBAL void* vpm, uint num_rows_in_block, uint num_blocks,
                  uint num_rows, uint num_cols) {
                GLOBAL TTRowPartitionedMatrix* pm =
                    (GLOBAL TTRowPartitionedMatrix*)vpm;
                pm->num_rows_in_block_ = num_rows_in_block;
                pm->num_blocks_ = num_blocks;
                pm->num_rows_ = num_rows;
                pm->num_cols_ = num_cols;
              }

              KERNEL void TTRowPartitionedMatrix_set(
                  GLOBAL void* vpm, GLOBAL void* block, uint idx) {
                GLOBAL TTRowPartitionedMatrix* pm =
                    (GLOBAL TTRowPartitionedMatrix*)vpm;
                pm->blocks_[idx] = (GLOBAL TT*)block;
              }

              KERNEL void TTRowPartitionedMatrixSizeof(GLOBAL ulong * size) {
            size[0] = sizeof(TTRowPartitionedMatrix);
          }

          )%%";
  return GetClTypes() + kSourceRowPartitionedMatrix;
}

inline std::string GetRowPartitionedMatrixHeader(const std::string& type) {
  return gen::MakeHeaderFromTemplate(type + "RowPartitionedMatrix",
                                     GetRowPartitionedMatrixSource(), "TT",
                                     type);
}

template <class T>
std::string GetRowPartitionedMatrixHeader() {
  return gen::MakeHeaderFromTemplate(type_name<T>() + "RowPartitionedMatrix",
                                     GetRowPartitionedMatrixSource(), "TT",
                                     type_name<T>());
}

template <class T>
class RowPartitionedMatrixFactory;

template <class T>
class RowPartitionedMatrix {
 public:
  uint32_t Rows() const { return rows_; }

  uint32_t Cols() const { return cols_; }

  uint32_t RowsPerBlock() const { return rows_per_alloc_; }

  std::vector<clcuda::Buffer<T>>& Blocks() { return blocks_; }

  clcuda::Buffer<char>& Get() { return base_; }

 private:
  RowPartitionedMatrix(clcuda::Queue queue, uint32_t rows, uint32_t cols,
                       uint32_t sizeOf, clcuda::Kernel init, clcuda::Kernel set,
                       uint32_t rowsInBlock = 0)
      : queue_(queue),
        rows_(rows),
        cols_(cols),
        rows_per_alloc_(rowsInBlock == 0
                            ? GetMaxRowsInBlock(queue.GetDevice(), cols_)
                            : rowsInBlock),
        base_(queue_.GetContext(), sizeOf) {
    for (uint32_t i = 0; i < rows_ / rows_per_alloc_; ++i) {
      blocks_.push_back(
          clcuda::Buffer<T>(queue_.GetContext(), rows_per_alloc_ * cols_));
    }
    if (rows_ % rows_per_alloc_) {
      blocks_.push_back(clcuda::Buffer<T>(queue_.GetContext(),
                                          (rows_ % rows_per_alloc_) * cols_));
    }
    init.SetArgument(0, base_);
    init.SetArgument(1, rows_per_alloc_);
    init.SetArgument(2, static_cast<uint32_t>(blocks_.size()));
    init.SetArgument(3, rows_);
    init.SetArgument(4, cols_);
    clcuda::Event e;
    init.Launch(queue_, {1}, {1}, e);
    queue_.Finish();
    set.SetArgument(0, base_);
    for (uint32_t i = 0; i < blocks_.size(); ++i) {
      set.SetArgument(1, blocks_[i]);
      set.SetArgument(2, i);
      set.Launch(queue, {1}, {1}, e);
      queue_.Finish();
    }
  }

  uint64_t GetMaxRowsInBlock(clcuda::Device dev, uint32_t cols) {
    // FIXME FIXME
    return 1024 * 1024;
#if 0
    uint64_t row_size = cols * sizeof(T);
    compute::ulong_ max_alloc =
        dev.get_info<compute::ulong_>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    return max_alloc / row_size;
#endif
  }

  clcuda::Queue queue_;
  uint32_t rows_;
  uint32_t cols_;
  uint32_t rows_per_alloc_;
  std::vector<clcuda::Buffer<T>> blocks_;
  clcuda::Buffer<char> base_;

  friend class RowPartitionedMatrixFactory<T>;
};

template <class T>
class RowPartitionedMatrixFactory
    : public std::enable_shared_from_this<RowPartitionedMatrixFactory<T>> {
 public:
  static std::shared_ptr<RowPartitionedMatrixFactory> New(clcuda::Queue queue) {
    return std::shared_ptr<RowPartitionedMatrixFactory>(
        new RowPartitionedMatrixFactory(queue));
  }

  RowPartitionedMatrix<T>* CreateMatrix(uint32_t rows, uint32_t cols,
                                        uint32_t rowsInBlock = 0) {
    RowPartitionedMatrix<T>* rpm = new RowPartitionedMatrix<T>(
        queue_, rows, cols, sizeOf_, *init_kernel_, *set_kernel_, rowsInBlock);
    return rpm;
  }

 private:
  RowPartitionedMatrixFactory(clcuda::Queue queue)
      : queue_(queue),
        prog_(queue_.GetContext(), GetRowPartitionedMatrixHeader<T>()) {
    std::vector<std::string> opts = ::mcmc::GetClFlags();
    clcuda::BuildStatus status = prog_.Build(queue_.GetDevice(), opts);
    LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
        << prog_.GetBuildInfo(queue_.GetDevice());
    clcuda::Buffer<uint64_t> sizeOf(queue_.GetContext(), 1);
    clcuda::Kernel sizeofKernel(prog_,
                                type_name<T>() + "RowPartitionedMatrixSizeof");
    sizeofKernel.SetArgument(0, sizeOf);
    clcuda::Event e;
    sizeofKernel.Launch(queue, {1}, {1}, e);
    queue_.Finish();
    sizeOf.Read(queue_, 1, &sizeOf_);
    init_kernel_.reset(new clcuda::Kernel(
        prog_, type_name<T>() + "RowPartitionedMatrix_init"));
    set_kernel_.reset(
        new clcuda::Kernel(prog_, type_name<T>() + "RowPartitionedMatrix_set"));
  }

  clcuda::Queue queue_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> init_kernel_;
  std::unique_ptr<clcuda::Kernel> set_kernel_;
  uint64_t sizeOf_;
};

}  // namespace mcmc

#endif  // __MCMC_PARTITIONED_ALLOC_H__
