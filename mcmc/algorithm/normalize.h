#ifndef __MCMC_ALGORITHM_NORMALIZE_H__
#define __MCMC_ALGORITHM_NORMALIZE_H__

#include <boost/compute/container/vector.hpp>
#include <glog/logging.h>
#include <string>

#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

namespace algorithm {

std::string WorkGroupNormalizeProgram(const std::string& type);

template <typename T>
class Normalizer {
 public:
  Normalizer(compute::command_queue queue, compute::vector<T>* in,
             uint32_t slice, uint32_t wg, compute::buffer* scratch = nullptr)
      : queue_(queue),
        data_(in),
        slice_(slice),
        wg_(wg),
        scratch_(scratch),
        scratch_is_owned_(scratch == nullptr) {
    LOG_IF(FATAL, data_->size() % slice != 0)
        << "Data size must be multiple of slice";
    uint32_t scratch_per_wg = slice / wg + (slice % wg ? 1 : 0);
    uint32_t scratch_size = (in->size() / slice + (in->size() % slice? 1: 0)) * scratch_per_wg * sizeof(T);
    if (scratch_is_owned_) {
      scratch_.reset(new compute::buffer(queue_.get_context(), scratch_size));
    } else {
      LOG_IF(FATAL, scratch_->size() < scratch_size)
          << "Need a larger scratch buffer";
    }
    prog_ = compute::program::create_with_source(
        WorkGroupNormalizeProgram(compute::type_name<T>()),
        queue_.get_context());
    try {
      prog_.build();
    } catch (compute::opencl_error& e) {
      LOG(FATAL) << prog_.build_log();
    }
    kernel_ = prog_.create_kernel(std::string("WG_NORMALIZE_KERNEL_") +
                                  std::string(compute::type_name<T>()));
    kernel_.set_arg(0, *data_);
    kernel_.set_arg(1, *scratch_);
    kernel_.set_arg(2, wg * sizeof(T), 0);
    kernel_.set_arg(3, static_cast<compute::uint_>(slice));
  }

  ~Normalizer() {
    if (!scratch_is_owned_) {
      scratch_.release();
    }
  }

  void operator()() {
    auto e = queue_.enqueue_1d_range_kernel(
        kernel_, 0, (data_->size() / slice_) * wg_, wg_);
    e.wait();
  }

 private:
  compute::command_queue queue_;
  compute::vector<T>* data_;
  uint32_t slice_;
  uint32_t wg_;
  std::unique_ptr<compute::buffer> scratch_;
  bool scratch_is_owned_;
  compute::program prog_;
  compute::kernel kernel_;
};

template <typename T>
class PartitionedNormalizer {
 public:
  PartitionedNormalizer(compute::command_queue queue,
                        RowPartitionedMatrix<T>* in,
                        compute::vector<Float>& sum, uint32_t wg,
                        compute::buffer* scratch = nullptr)
      : queue_(queue),
        data_(in),
        sum_(sum),
        wg_(wg),
        scratch_(scratch),
        scratch_is_owned_(scratch == nullptr) {
    uint32_t scratch_per_wg = data_->Cols() / wg + (data_->Cols() % wg ? 1 : 0);
    uint32_t scratch_size = sizeof(T) * scratch_per_wg * in->Rows();
    if (scratch_is_owned_) {
      scratch_.reset(new compute::buffer(queue_.get_context(), scratch_size));
    } else {
      LOG_IF(FATAL, scratch_->size() < scratch_size)
          << "Need a larger scratch buffer";
    }
    prog_ = compute::program::create_with_source(
        WorkGroupNormalizeProgram(compute::type_name<T>()),
        queue_.get_context());
    try {
      prog_.build();
    } catch (compute::opencl_error& e) {
      LOG(FATAL) << prog_.build_log();
    }
    kernel_ =
        prog_.create_kernel(std::string("WG_NORMALIZE_PARTITIONED_KERNEL_") +
                            std::string(compute::type_name<T>()));
    kernel_.set_arg(0, data_->Get());
    kernel_.set_arg(1, sum_);
    kernel_.set_arg(2, *scratch_);
    kernel_.set_arg(3, wg * sizeof(T), 0);
  }

  ~PartitionedNormalizer() {
    if (!scratch_is_owned_) {
      scratch_.release();
    }
  }

  void operator()() {
    auto e =
        queue_.enqueue_1d_range_kernel(kernel_, 0, data_->Rows() * wg_, wg_);
    e.wait();
  }

 private:
  compute::command_queue queue_;
  RowPartitionedMatrix<T>* data_;
  compute::vector<Float>& sum_;
  uint32_t wg_;
  std::unique_ptr<compute::buffer> scratch_;
  bool scratch_is_owned_;
  compute::program prog_;
  compute::kernel kernel_;
};

}  // namespace algorithm
}  // namespace mcmc

#endif  // __MCMC_ALGORITHM_NORMALIZE_H__
