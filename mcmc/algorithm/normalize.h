#ifndef __MCMC_ALGORITHM_NORMALIZE_H__
#define __MCMC_ALGORITHM_NORMALIZE_H__

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
  Normalizer(clcuda::Queue queue, clcuda::Buffer<T>* in, uint32_t slice,
             uint32_t wg)
      : queue_(queue),
        data_(in),
        slice_(slice),
        wg_(wg),
        prog_(queue_.GetContext(), WorkGroupNormalizeProgram(type_name<T>())) {
    LOG_IF(FATAL, (data_->GetSize() / sizeof(T)) % slice != 0)
        << "Data size must be multiple of slice";
    std::vector<std::string> opts = ::mcmc::GetClFlags(wg_);
    clcuda::BuildStatus status = prog_.Build(queue_.GetDevice(), opts);
    LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
        << prog_.GetBuildInfo(queue_.GetDevice());
    kernel_.reset(new clcuda::Kernel(
        prog_, std::string("WG_NORMALIZE_KERNEL_") + type_name<T>()));
    kernel_->SetArgument(0, *data_);
    kernel_->SetArgument(1, static_cast<uint32_t>((data_->GetSize() / sizeof(T)) / slice));
    kernel_->SetArgument(2, static_cast<uint32_t>(slice));
  }

  void operator()() {
    clcuda::Event e;
    uint32_t num_groups = (data_->GetSize() / sizeof(T)) / slice_;
    num_groups = std::min(num_groups, GetMaxGroups());
    kernel_->Launch(queue_, {num_groups * wg_},
                    {wg_}, e);
    queue_.Finish();
  }

 private:
  clcuda::Queue queue_;
  clcuda::Buffer<T>* data_;
  uint32_t slice_;
  uint32_t wg_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> kernel_;
};

template <typename T>
class PartitionedNormalizer {
 public:
  PartitionedNormalizer(clcuda::Queue queue, RowPartitionedMatrix<T>* in,
                        clcuda::Buffer<Float>& sum, uint32_t wg)
      : queue_(queue),
        data_(in),
        sum_(sum),
        wg_(wg),
        prog_(queue_.GetContext(), WorkGroupNormalizeProgram(type_name<T>())) {
    std::vector<std::string> opts = ::mcmc::GetClFlags(wg_);
    clcuda::BuildStatus status = prog_.Build(queue_.GetDevice(), opts);
    LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
        << prog_.GetBuildInfo(queue_.GetDevice());
    kernel_.reset(new clcuda::Kernel(
        prog_,
        std::string("WG_NORMALIZE_PARTITIONED_KERNEL_") + type_name<T>()));
    kernel_->SetArgument(0, data_->Get());
    kernel_->SetArgument(1, sum_);
  }

  void operator()() {
    clcuda::Event e;
    kernel_->Launch(queue_, {std::min(data_->Rows(), GetMaxGroups()) * wg_}, {wg_}, e);
    queue_.Finish();
  }

 private:
  clcuda::Queue queue_;
  RowPartitionedMatrix<T>* data_;
  clcuda::Buffer<Float>& sum_;
  uint32_t wg_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> kernel_;
};

}  // namespace algorithm
}  // namespace mcmc

#endif  // __MCMC_ALGORITHM_NORMALIZE_H__
