#ifndef __MCMC_RANDOM_H__
#define __MCMC_RANDOM_H__

#include <memory>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

namespace mcmc {
namespace random {

namespace compute = boost::compute;

typedef compute::ulong2_ gsl_rng;
typedef gsl_rng random_seed_t;

const std::string kClRandomTypes = BOOST_COMPUTE_STRINGIZE_SOURCE(

    typedef ulong2 gsl_rng; typedef gsl_rng random_seed_t;

    typedef struct {
      __global random_seed_t* base_;
      ulong num_seeds;
    } Random;

    );

#include "mcmc/random.cl.inc"

class OpenClRandomFactory;

class OpenClRandom {
 public:
  inline compute::buffer& Get() { return buf_; }

 private:
  OpenClRandom(std::shared_ptr<OpenClRandomFactory> factory,
               compute::kernel* init, compute::command_queue* queue,
               uint64_t sizeOfRandom, uint64_t size, random_seed_t seed);

  std::shared_ptr<OpenClRandomFactory> factory_;
  compute::command_queue queue_;
  compute::vector<random_seed_t> data_;
  compute::buffer buf_;

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

}  // namespapce random
}  // namespace mcmc

#endif  // __MCMC_RANDOM_H__
