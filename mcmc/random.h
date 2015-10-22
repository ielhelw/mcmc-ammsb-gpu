#ifndef __MCMC_RANDOM_H__
#define __MCMC_RANDOM_H__

#include <memory>

#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {
namespace random {

typedef ::mcmc::ulong2 gsl_rng;
typedef gsl_rng random_seed_t;

class OpenClRandomFactory;

class OpenClRandom {
 public:
  inline clcuda::Buffer<char>& Get() { return buf_; }
  inline clcuda::Buffer<random_seed_t>& GetSeeds() { return data_; }

 private:
  OpenClRandom(std::shared_ptr<OpenClRandomFactory> factory,
               clcuda::Kernel* init, clcuda::Queue* queue,
               uint64_t sizeOfRandom, uint64_t size, random_seed_t seed);

  void SetSeed(random_seed_t seed);

  std::shared_ptr<OpenClRandomFactory> factory_;
  clcuda::Queue queue_;
  clcuda::Buffer<random_seed_t> data_;
  clcuda::Buffer<char> buf_;
  clcuda::Kernel init_kernel_;

  friend class OpenClRandomFactory;
};

class OpenClRandomFactory
    : public std::enable_shared_from_this<OpenClRandomFactory> {
 public:
  static std::shared_ptr<OpenClRandomFactory> New(clcuda::Queue queue);

  OpenClRandom* CreateRandom(uint64_t size, random_seed_t seed);

 private:
  OpenClRandomFactory(clcuda::Queue queue);

  clcuda::Queue queue_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> init_kernel_;
  uint64_t sizeOfRandom_;
};

const std::string GetRandomTypes();
const std::string GetRandomHeader();

void RandomGamma(clcuda::Queue* queue, OpenClRandom* randv, Float eta0,
                 Float eta1, RowPartitionedMatrix<Float>* norm);

void RandomGammaAndNormalize(clcuda::Queue* queue, Float eta0, Float eta1,
                             RowPartitionedMatrix<Float>* norm,
                             clcuda::Buffer<Float>* sum);

template <class Generator>
void RandomAndNormalize(clcuda::Queue* queue, Generator* gen,
                        clcuda::Buffer<Float>* base,
                        clcuda::Buffer<Float>* norm, uint32_t cols) {
  std::vector<Float> host_base(base->GetSize() / sizeof(Float));
  std::generate(host_base.begin(), host_base.end(), *gen);
  base->Write(*queue, host_base.size(), host_base.data());
  norm->Write(*queue, host_base.size(), host_base.data());
  mcmc::algorithm::Normalizer<Float>(*queue, norm, cols, 1)();
}

}  // namespapce random
}  // namespace mcmc

#endif  // __MCMC_RANDOM_H__
