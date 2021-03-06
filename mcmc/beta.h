#ifndef __MCMC_BETA_H__
#define __MCMC_BETA_H__

#include "mcmc/config.h"
#include "mcmc/random.h"
#include "mcmc/algorithm/normalize.h"
#include "mcmc/serialize.h"

bool Serialize(std::ostream* out);

bool Parse(std::istream* in);

namespace mcmc {

class BetaUpdater {
 public:
  enum Mode { EDGE_PER_THREAD, EDGE_PER_WORKGROUP };

  BetaUpdater(Mode mode, const Config& cfg, clcuda::Queue queue,
              clcuda::Buffer<Float>& theta, clcuda::Buffer<Float>& beta,
              RowPartitionedMatrix<Float>* pi, OpenClSet* trainingSet,
              const std::vector<std::string>& compileFlags,
              const std::string& baseFuncs);

  void operator()(clcuda::Buffer<Edge>* edges, uint32_t num_edges, Float scale);

  clcuda::Buffer<Float>& GetThetaSum() { return theta_sum_; }
  clcuda::Buffer<Float>& GetGrads() { return grads_; }

  double ThetaSumTime() const { return t_theta_sum_; }
  double GradsPartialTime() const { return t_grads_partial_; }
  double GradsSumTime() const { return t_grads_sum_; }
  double UpdateThetaTime() const { return t_update_theta_; }
  double NormalizeTime() const { return t_normalize_; }

  bool Serialize(std::ostream* out);

  bool Parse(std::istream* in);

 private:
  Mode mode_;
  clcuda::Queue queue_;

  clcuda::Buffer<Float>& theta_;     // [K, 2]
  clcuda::Buffer<Float>& beta_;      // [K, 2]
  RowPartitionedMatrix<Float>* pi_;  // [N, K]
  OpenClSet* trainingSet_;

  std::unique_ptr<clcuda::Program> prog_;
  std::unique_ptr<clcuda::Kernel> theta_sum_kernel_;
  std::unique_ptr<clcuda::Kernel> grads_partial_kernel_;
  std::unique_ptr<clcuda::Kernel> grads_sum_kernel_;
  std::unique_ptr<clcuda::Kernel> update_theta_kernel_;
  algorithm::Normalizer<Float> normalizer_;

  std::shared_ptr<random::OpenClRandomFactory> randFactory_;
  std::unique_ptr<random::OpenClRandom> rand_;

  uint32_t count_calls_;
  uint32_t k_;
  uint32_t local_;

  clcuda::Buffer<Float> theta_sum_;                 // [K]
  clcuda::Buffer<Float> grads_;                     // [K, 2]
  std::unique_ptr<clcuda::Buffer<Float>> probs_;                     // [mini_batch_edges, K]

  double t_theta_sum_;
  double t_grads_partial_;
  double t_grads_sum_;
  double t_update_theta_;
  double t_normalize_;
};

}  // namespace mcmc
#endif  // __MCMC_BETA_H__
