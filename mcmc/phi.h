#ifndef __MCMC_PHI_H__
#define __MCMC_PHI_H__

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>

#include "mcmc/config.h"
#include "mcmc/random.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

class PhiUpdater {
 public:
  enum Mode { NODE_PER_THREAD, NODE_PER_WORKGROUP };

  PhiUpdater(Mode mode, const Config& cfg, compute::command_queue queue,
             compute::vector<Float>& beta, RowPartitionedMatrix<Float>* pi,
             compute::vector<Float>& phi, OpenClSet* trainingSet,
             const std::string& compileFlags, const std::string& baseFuncs);

  void operator()(
      compute::vector<Vertex>& mini_batch_nodes,  // [X <= 2*MINI_BATCH_SIZE]
      compute::vector<Vertex>& neighbors,  // [MINI_BATCH_NODES, NUM_NEIGHBORS]
      uint32_t num_mini_batch_nodes);

  uint64_t LastInvocationTime() const;

 private:
  Mode mode_;
  compute::command_queue queue_;
  compute::event phi_event_;
  compute::event pi_event_;

  compute::vector<Float>& beta_;     // [K]
  RowPartitionedMatrix<Float>* pi_;  // [N,K]
  compute::vector<Float>& phi_;      // [N]
  compute::vector<Float> phi_vec;    // [2*mini_batch, K]
  OpenClSet* trainingSet_;

  std::shared_ptr<random::OpenClRandomFactory> randFactory_;
  std::unique_ptr<random::OpenClRandom> rand_;

  compute::program prog_;
  compute::kernel phi_kernel_;
  compute::kernel pi_kernel_;

  uint32_t count_calls_;
  uint32_t k_;
  uint32_t local_;

  compute::vector<Float> grads_;    // [mini_batch, K]
  compute::vector<Float> probs_;    // [mini_batch, K]
  compute::vector<Float> scratch_;  // [mini_batch, K]
};

}  // namespace mcmc

#endif  // __MCMC_PHI_H__
