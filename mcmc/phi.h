#ifndef __MCMC_PHI_H__
#define __MCMC_PHI_H__

#include "mcmc/config.h"
#include "mcmc/random.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

class PhiUpdater {
 public:
  enum Mode {
    NODE_PER_THREAD,
    NODE_PER_WORKGROUP
  };

  PhiUpdater(Mode mode, const Config& cfg, clcuda::Queue queue,
             clcuda::Buffer<Float>& beta, RowPartitionedMatrix<Float>* pi,
             clcuda::Buffer<Float>& phi, OpenClSet* trainingSet,
             const std::vector<std::string>& compileFlags,
             const std::string& baseFuncs);

  void operator()(
      clcuda::Buffer<Vertex>& mini_batch_nodes,  // [X <= 2*MINI_BATCH_SIZE]
      clcuda::Buffer<Vertex>& neighbors,  // [MINI_BATCH_NODES, NUM_NEIGHBORS]
      uint32_t num_mini_batch_nodes);

  uint64_t LastInvocationTime() const;

 private:
  Mode mode_;
  clcuda::Queue queue_;
  clcuda::Event phi_event_;
  clcuda::Event pi_event_;

  clcuda::Buffer<Float>& beta_;      // [K]
  RowPartitionedMatrix<Float>* pi_;  // [N,K]
  clcuda::Buffer<Float>& phi_;       // [N]
  clcuda::Buffer<Float> phi_vec;     // [2*mini_batch, K]
  OpenClSet* trainingSet_;

  std::shared_ptr<random::OpenClRandomFactory> randFactory_;
  std::unique_ptr<random::OpenClRandom> rand_;

  std::unique_ptr<clcuda::Program> prog_;
  std::unique_ptr<clcuda::Kernel> phi_kernel_;
  std::unique_ptr<clcuda::Kernel> pi_kernel_;

  uint32_t count_calls_;
  uint32_t k_;
  uint32_t local_;

  std::unique_ptr<clcuda::Buffer<Float>> grads_;    // [mini_batch, K]
  std::unique_ptr<clcuda::Buffer<Float>> probs_;    // [mini_batch, K]
};

}  // namespace mcmc

#endif  // __MCMC_PHI_H__
