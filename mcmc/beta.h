#ifndef __MCMC_BETA_H__
#define __MCMC_BETA_H__

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>

#include "mcmc/config.h"
#include "mcmc/random.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

class BetaUpdater {
 public:
  enum Mode { EDGE_PER_THREAD, EDGE_PER_WORKGROUP };

  BetaUpdater(Mode mode, const Config& cfg, compute::command_queue queue,
              compute::vector<Float>& theta, compute::vector<Float>& beta,
              compute::vector<Float>& pi, OpenClSet* trainingSet,
              const std::string& compileFlags, const std::string& baseFuncs);

  void operator()(compute::vector<Edge>* edges, uint32_t num_edges,
                  Float scale);

  uint64_t LastInvocationTime() const;

  const compute::vector<Float>& GetThetaSum() { return theta_sum_; }
  const compute::vector<Float>& GetGrads() { return grads_; }

 private:
  Mode mode_;
  compute::command_queue queue_;

  compute::vector<Float>& theta_;  // [K, 2]
  compute::vector<Float>& beta_;   // [K]
  compute::vector<Float>& pi_;     // [N, K]
  OpenClSet* trainingSet_;

  compute::program prog_;
  compute::kernel theta_sum_kernel_;
  compute::kernel grads_partial_kernel_;
  compute::kernel grads_sum_kernel_;
  compute::kernel update_theta_kernel_;
  compute::kernel beta_kernel_;
  algorithm::Normalizer<Float> normalizer_;


  std::shared_ptr<random::OpenClRandomFactory> randFactory_;
  std::unique_ptr<random::OpenClRandom> rand_;

  uint32_t count_calls_;
  uint32_t k_;
  uint32_t local_;

  compute::vector<Float> theta_sum_;  // [K]
  compute::vector<Float> grads_;      // [K, 2]
  compute::vector<Float> probs_;      // [mini_batch_edges, K]
  compute::vector<Float> scratch_;    // [mini_batch_edges, K]

  compute::event theta_sum_event_;
  compute::event grads_partial_event_;
  compute::event grads_sum_event_;
  compute::event update_theta_event_;
  uint64_t normalize_time_;
};

}  // namespace mcmc
#endif  // __MCMC_BETA_H__
