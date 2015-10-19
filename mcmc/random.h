#ifndef __MCMC_RANDOM_H__
#define __MCMC_RANDOM_H__

#include <memory>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {
namespace random {

typedef compute::ulong2_ gsl_rng;
typedef gsl_rng random_seed_t;

class OpenClRandomFactory;

class OpenClRandom {
 public:
  inline compute::buffer& Get() { return buf_; }
  inline compute::vector<random_seed_t>& GetSeeds() { return data_; }

 private:
  OpenClRandom(std::shared_ptr<OpenClRandomFactory> factory,
               compute::kernel* init, compute::command_queue* queue,
               uint64_t sizeOfRandom, uint64_t size, random_seed_t seed);

  void SetSeed(random_seed_t seed);

  std::shared_ptr<OpenClRandomFactory> factory_;
  compute::command_queue queue_;
  compute::vector<random_seed_t> data_;
  compute::buffer buf_;
  compute::kernel init_kernel_;

  friend class OpenClRandomFactory;
};

class OpenClRandomFactory
    : public std::enable_shared_from_this<OpenClRandomFactory> {
 public:
  static std::shared_ptr<OpenClRandomFactory> New(compute::command_queue queue);

  OpenClRandom* CreateRandom(uint64_t size, random_seed_t seed);

 private:
  OpenClRandomFactory(compute::command_queue queue);

  compute::program prog_;
  compute::command_queue queue_;
  compute::kernel init_kernel_;
  uint64_t sizeOfRandom_;
};

const std::string GetRandomTypes();
const std::string GetRandomHeader();

void RandomGamma(compute::command_queue* queue, OpenClRandom* randv, Float eta0,
                 Float eta1, RowPartitionedMatrix<Float>* norm);

void RandomGammaAndNormalize(compute::command_queue* queue,
                             Float eta0, Float eta1,
                             RowPartitionedMatrix<Float>* norm,
                             compute::vector<Float>* sum);

template <class Generator>
void RandomAndNormalize(compute::command_queue* queue, Generator* gen,
                             compute::vector<Float>* base,
                             compute::vector<Float>* norm, uint32_t cols) {
  std::vector<Float> host_base(base->size());
  std::generate(host_base.begin(), host_base.end(), *gen);
  compute::copy(host_base.begin(), host_base.end(), base->begin(), *queue);
  compute::copy(base->begin(), base->end(), norm->begin(), *queue);
  mcmc::algorithm::Normalizer<Float>(*queue, norm, cols, 1)();
}

}  // namespapce random
}  // namespace mcmc

#endif  // __MCMC_RANDOM_H__
