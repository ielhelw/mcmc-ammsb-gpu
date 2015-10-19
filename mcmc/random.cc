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
          __kernel void SizeOfRandom(__global ulong *
                                     size) { *size = sizeof(Random); }

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
           compute::memory_object::read_write),
      init_kernel_(*init) {
  SetSeed(seed);
}

void OpenClRandom::SetSeed(random_seed_t seed) {
  init_kernel_.set_arg(0, buf_);
  init_kernel_.set_arg(1, static_cast<compute::int_>(data_.size()));
  init_kernel_.set_arg(2, seed);
  init_kernel_.set_arg(3, data_);
  auto e = queue_.enqueue_task(init_kernel_);
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

const std::string& GenerateGammaSource() {
  static const std::string kSource =
      BOOST_COMPUTE_STRINGIZE_SOURCE(__kernel void generate_gamma(
          __global void * vpi, __global void * vrand, TT a, TT b) {
        __global TTRowPartitionedMatrix* pm =
            (__global TTRowPartitionedMatrix*)vpi;
        uint i = get_group_id(0);
        uint gsize = get_num_groups(0);
        uint lid = get_local_id(0);
        uint lsize = get_local_size(0);
        __global Random* seeds = (__global Random*)vrand;

        if (i < pm->num_rows_) {
          random_seed_t seed = seeds->base_[get_global_id(0)];
          for (; i < pm->num_rows_; i += gsize) {
            __global TT* row = TTRowPartitionedMatrix_Row(pm, i);
            for (uint j = lid; j < pm->num_cols_; j += lsize) {
              row[j] = rand_gamma(&seed, a, b);
            }
          }
          seeds->base_[get_global_id(0)] = seed;
        }
      });
  return kSource;
}

void RandomGamma(compute::command_queue* queue, OpenClRandom* randv, Float eta0,
                 Float eta1, RowPartitionedMatrix<Float>* norm) {
  uint32_t local = 32;
  LOG_IF(FATAL, randv->GetSeeds().size() < local * norm->Rows())
      << "RandomSeeds vector too small";
  std::ostringstream out;
  out << random::GetRandomHeader() << std::endl;
  out << GetRowPartitionedMatrixHeader<Float>() << std::endl;
  out << GenerateGammaSource() << std::endl;
  std::string source = gen::MakeHeaderFromTemplate(
      "GenAndNorm", out.str(), "TT", compute::type_name<Float>());
  compute::program prog =
      compute::program::create_with_source(source, queue->get_context());
  try {
    prog.build();
  }
  catch (compute::opencl_error& e) {
    LOG(FATAL) << prog.build_log();
  }
  compute::kernel kernel = prog.create_kernel("generate_gamma");
  kernel.set_arg(0, norm->Get());
  kernel.set_arg(1, randv->Get());
  kernel.set_arg(2, eta0);
  kernel.set_arg(3, eta1);
  queue->enqueue_1d_range_kernel(kernel, 0, norm->Rows() * local, local).wait();
}

void RandomGammaAndNormalize(compute::command_queue* queue,
                             Float eta0, Float eta1,
                             RowPartitionedMatrix<Float>* norm,
                             compute::vector<Float>* sum) {
  auto randFactory = random::OpenClRandomFactory::New(*queue);
  std::unique_ptr<random::OpenClRandom>
    randv(randFactory->CreateRandom(norm->Rows() * 32,
          random::random_seed_t{11, 113}));
  random::RandomGamma(queue, randv.get(), eta0, eta1, norm);
  mcmc::algorithm::PartitionedNormalizer<Float>(*queue, norm, *sum, 32)();
}

}  // namespapce random
}  // namespace mcmc
