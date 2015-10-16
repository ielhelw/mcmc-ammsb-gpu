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
             uint32_t slice, uint32_t wg)
      : queue_(queue),
        data_(in),
        slice_(slice),
        wg_(wg) {
    LOG_IF(FATAL, data_->size() % slice != 0)
        << "Data size must be multiple of slice";
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
    kernel_.set_arg(1, wg * sizeof(T), 0);
    kernel_.set_arg(2, static_cast<compute::uint_>(slice));
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
  compute::program prog_;
  compute::kernel kernel_;
};

template <typename T>
class PartitionedNormalizer {
 public:
  PartitionedNormalizer(compute::command_queue queue,
                        RowPartitionedMatrix<T>* in,
                        compute::vector<Float>& sum, uint32_t wg)
      : queue_(queue),
        data_(in),
        sum_(sum),
        wg_(wg) {
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
    kernel_.set_arg(2, wg * sizeof(T), 0);
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
  compute::program prog_;
  compute::kernel kernel_;
};

}  // namespace algorithm
}  // namespace mcmc

#endif  // __MCMC_ALGORITHM_NORMALIZE_H__
