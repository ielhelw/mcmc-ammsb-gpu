#include "mcmc/random.h"
#include <glog/logging.h>

#include "mcmc/gen-util.h"

namespace mcmc {
namespace random {

#include "mcmc/random.cl.inc"

const std::string GetRandomTypes() {
  static const std::string kClRandomTypes = R"%%(

      typedef ulong2 gsl_rng; typedef gsl_rng random_seed_t;

      typedef struct {
        GLOBAL random_seed_t* base_;
        ulong num_seeds;
      } Random;

      )%%";
  return ::mcmc::gen::MakeHeader("RANDOM_TYPES", kClRandomTypes);
}

const std::string GetRandomSource() {
  static const std::string kClRandomSource =
      ::mcmc::GetClTypes() + GetRandomTypes() + R"%%(
          KERNEL void SizeOfRandom(GLOBAL ulong *
                                   size) { *size = sizeof(Random); }

          KERNEL void RandomInit(GLOBAL void* base, int num_random_seeds,
                                 random_seed_t random_seed,
                                 GLOBAL random_seed_t* thread_random_seed) {
            size_t id = GET_GLOBAL_ID();
            for (size_t i = id; i < num_random_seeds; i += GET_GLOBAL_SIZE()) {
              thread_random_seed[i].x = random_seed.x + i;
              thread_random_seed[i].y = random_seed.y + i;
            }
            if (id == 0) {
              GLOBAL Random* random = (GLOBAL Random*)base;
              random->base_ = thread_random_seed;
              random->num_seeds = num_random_seeds;
            }
          })%%";
  return ::mcmc::gen::MakeHeader("RANDOM_SOURCE", kClRandomSource);
}

OpenClRandom::OpenClRandom(std::shared_ptr<OpenClRandomFactory> factory,
                           clcuda::Kernel* init, clcuda::Queue* queue,
                           uint64_t sizeOfRandom, uint64_t size,
                           random_seed_t seed)
    : factory_(factory),
      queue_(*queue),
      data_(queue_.GetContext(), size),
      buf_(queue_.GetContext(), sizeOfRandom),
      init_kernel_(*init) {
  SetSeed(seed);
}

void OpenClRandom::SetSeed(random_seed_t seed) {
  init_kernel_.SetArgument(0, buf_);
  init_kernel_.SetArgument(
      1, static_cast<int32_t>(data_.GetSize() / sizeof(random_seed_t)));
  init_kernel_.SetArgument(2, seed);
  init_kernel_.SetArgument(3, data_);
  clcuda::Event e;
  init_kernel_.Launch(queue_, {1}, {1}, e);
  queue_.Finish();
}

std::shared_ptr<OpenClRandomFactory> OpenClRandomFactory::New(
    clcuda::Queue queue) {
  return std::shared_ptr<OpenClRandomFactory>(new OpenClRandomFactory(queue));
}

OpenClRandom* OpenClRandomFactory::CreateRandom(uint64_t size,
                                                random_seed_t seed) {
  return new OpenClRandom(shared_from_this(), init_kernel_.get(), &queue_,
                          sizeOfRandom_, size, seed);
}

OpenClRandomFactory::OpenClRandomFactory(clcuda::Queue queue)
    : queue_(queue), prog_(queue_.GetContext(), GetRandomSource()) {
  std::vector<std::string> opts = ::mcmc::GetClFlags();
  clcuda::BuildStatus status = prog_.Build(queue_.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_.GetBuildInfo(queue_.GetDevice());
  init_kernel_.reset(new clcuda::Kernel(prog_, "RandomInit"));
  clcuda::Kernel sizeOf_kernel(prog_, "SizeOfRandom");
  clcuda::Buffer<uint64_t> size(queue_.GetContext(), 1);
  sizeOf_kernel.SetArgument(0, size);
  clcuda::Event e;
  sizeOf_kernel.Launch(queue_, {1}, {1}, e);
  queue_.Finish();
  size.Read(queue_, 1, &sizeOfRandom_);
}

std::string GenerateGammaSource() {
  static const std::string kSource = R"%%(
      KERNEL void generate_gamma(
          GLOBAL void* vpi, GLOBAL void* vrand, TT a, TT b) {
        GLOBAL TTRowPartitionedMatrix* pm = (GLOBAL TTRowPartitionedMatrix*)vpi;
        uint i = GET_GROUP_ID();
        uint gsize = GET_NUM_GROUPS();
        uint lid = GET_LOCAL_ID();
        uint lsize = GET_LOCAL_SIZE();
        GLOBAL Random* seeds = (GLOBAL Random*)vrand;

        if (i < pm->num_rows_) {
          random_seed_t seed = seeds->base_[GET_GLOBAL_ID()];
          for (; i < pm->num_rows_; i += gsize) {
            GLOBAL TT* row = TTRowPartitionedMatrix_Row(pm, i);
            for (uint j = lid; j < pm->num_cols_; j += lsize) {
              row[j] = rand_gamma(&seed, a, b);
            }
          }
          seeds->base_[GET_GLOBAL_ID()] = seed;
        }
      })%%";
  return ::mcmc::GetClTypes() + kSource;
}

void RandomGamma(clcuda::Queue* queue, OpenClRandom* randv, Float eta0,
                 Float eta1, RowPartitionedMatrix<Float>* norm) {
  uint32_t local = 32;
  LOG_IF(FATAL, randv->GetSeeds().GetSize() / sizeof(random_seed_t) <
                    local * norm->Rows())
      << "RandomSeeds vector too small";
  std::ostringstream out;
  out << random::GetRandomHeader() << std::endl;
  out << GetRowPartitionedMatrixHeader<Float>() << std::endl;
  out << GenerateGammaSource() << std::endl;
  std::string source = gen::MakeHeaderFromTemplate(
      "GenAndNorm", out.str(), "TT", compute::type_name<Float>());
  clcuda::Program prog(queue->GetContext(), source);
  std::vector<std::string> opts = ::mcmc::GetClFlags();
  clcuda::BuildStatus status = prog.Build(queue->GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog.GetBuildInfo(queue->GetDevice());
  clcuda::Kernel kernel(prog, "generate_gamma");
  kernel.SetArgument(0, norm->Get());
  kernel.SetArgument(1, randv->Get());
  kernel.SetArgument(2, eta0);
  kernel.SetArgument(3, eta1);
  clcuda::Event e;
  kernel.Launch(*queue, {norm->Rows() * local}, {local}, e);
  queue->Finish();
}

void RandomGammaAndNormalize(clcuda::Queue* queue, Float eta0, Float eta1,
                             RowPartitionedMatrix<Float>* norm,
                             clcuda::Buffer<Float>* sum) {
  auto randFactory = random::OpenClRandomFactory::New(*queue);
  std::unique_ptr<random::OpenClRandom> randv(randFactory->CreateRandom(
      norm->Rows() * 32, random::random_seed_t{11, 113}));
  random::RandomGamma(queue, randv.get(), eta0, eta1, norm);
  mcmc::algorithm::PartitionedNormalizer<Float>(*queue, norm, *sum, 32)();
}

}  // namespapce random
}  // namespace mcmc
