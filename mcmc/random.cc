#include "mcmc/random.h"
#include <glog/logging.h>

#include "mcmc/gen-util.h"

namespace mcmc {
namespace random {

#include "mcmc/random.cl.inc"

const std::string GetRandomTypes() {
  static const std::string kClRandomTypes = BOOST_COMPUTE_STRINGIZE_SOURCE(

    typedef ulong2 gsl_rng; typedef gsl_rng random_seed_t;

    typedef struct {
      __global random_seed_t* base_;
      ulong num_seeds;
    } Random;

    );
  return ::mcmc::gen::MakeHeader("RANDOM_TYPES", kClRandomTypes);
}

const std::string GetRandomSource() {
  static const std::string kClRandomSource =
    GetRandomTypes() +
    BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void SizeOfRandom(__global ulong* size) {
          *size = sizeof(Random);
        }

        __kernel void RandomInit(__global void* base, int num_random_seeds,
                                 random_seed_t random_seed,
                                 __global random_seed_t* thread_random_seed) {
          size_t id = get_global_id(0);
          for (size_t i = id; i < num_random_seeds; i += get_global_size(0)) {
            thread_random_seed[i].x = random_seed.x + i;
            thread_random_seed[i].y = random_seed.y + i;
          }
          if (id == 0) {
            __global Random* random = (__global Random*)base;
            random->base_ = thread_random_seed;
            random->num_seeds = num_random_seeds;
          }
        });
  return ::mcmc::gen::MakeHeader("RANDOM_SOURCE", kClRandomSource);
}

OpenClRandom::OpenClRandom(std::shared_ptr<OpenClRandomFactory> factory,
                           compute::kernel* init, compute::command_queue* queue,
                           uint64_t sizeOfRandom, uint64_t size,
                           random_seed_t seed)
    : factory_(factory),
      queue_(*queue),
      data_(size, queue_.get_context()),
      buf_(queue_.get_context(), sizeOfRandom,
           compute::memory_object::read_write) {
  init->set_arg(0, buf_);
  init->set_arg(1, static_cast<compute::int_>(size));
  init->set_arg(2, seed);
  init->set_arg(3, data_);
  auto e = queue_.enqueue_task(*init);
  e.wait();
}

std::shared_ptr<OpenClRandomFactory> OpenClRandomFactory::New(
    compute::command_queue queue) {
  return std::shared_ptr<OpenClRandomFactory>(new OpenClRandomFactory(queue));
}

OpenClRandom* OpenClRandomFactory::CreateRandom(uint64_t size,
                                                random_seed_t seed) {
  return new OpenClRandom(shared_from_this(), &init_kernel_, &queue_,
                          sizeOfRandom_, size, seed);
}

OpenClRandomFactory::OpenClRandomFactory(compute::command_queue queue)
    : queue_(queue) {
  prog_ = compute::program::create_with_source(GetRandomSource(),
                                               queue_.get_context());
  try {
    prog_.build();
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
    ;
  }
  init_kernel_ = prog_.create_kernel("RandomInit");
  compute::kernel sizeOf_kernel = prog_.create_kernel("SizeOfRandom");
  compute::vector<uint64_t> size(1, (uint64_t)0, queue_);
  sizeOf_kernel.set_arg(0, size);
  auto e = queue_.enqueue_task(sizeOf_kernel);
  e.wait();
  compute::copy(size.begin(), size.end(), &sizeOfRandom_, queue_);
}

}  // namespapce random
}  // namespace mcmc
