#ifndef __MCMC_LEARNER_H__
#define __MCMC_LEARNER_H__

#include <boost/program_options.hpp>
#include <future>
#include <ostream>
#include <signal.h>

#include "mcmc/algorithm/normalize.h"
#include "mcmc/config.h"
#include "mcmc/data.h"
#include "mcmc/perplexity.h"
#include "mcmc/phi.h"
#include "mcmc/beta.h"

namespace mcmc {

class Learner {
 public:
  Learner(const Config& cfg, clcuda::Queue queue);

  void Run(uint32_t max_iters, sig_atomic_t* signaled = nullptr);

  Float HeldoutPerplexity();

  Float TrainingPerplexity();

  void PrintStats();

  bool Serialize(std::ostream* out);

  bool Parse(std::istream* in);

  static const std::string GetBaseFuncs();

 private:
  Float SampleMiniBatch(std::vector<Edge>* edges, unsigned int* seed);

  void ExtractNodesFromMiniBatch(const std::vector<Edge>& edges,
                                 std::vector<Vertex>* nodes);

  Float DoSample(Sample* sample);

  const Config& cfg_;

  clcuda::Queue queue_;
  clcuda::Buffer<Float> beta_;   // [K]
  clcuda::Buffer<Float> theta_;  // [K]

  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory_;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;  // [N,K]
  clcuda::Buffer<Float> phi_;                        // [N]

  std::shared_ptr<OpenClSetFactory> setFactory_;

  std::unique_ptr<OpenClSet> trainingSet_;
  std::unique_ptr<OpenClSet> heldoutSet_;
  clcuda::Buffer<Edge> trainingEdges_;
  clcuda::Buffer<Edge> heldoutEdges_;

  std::vector<std::string> compileFlags_;

#ifdef MCMC_CALC_TRAIN_PPX
  std::vector<Edge> trainingPerplexityEdges_;
  clcuda::Buffer<Edge> devTrainingPerplexityEdges_;
  PerplexityCalculator trainingPerplexity_;
#endif

  PerplexityCalculator heldoutPerplexity_;
  PhiUpdater phiUpdater_;
  BetaUpdater betaUpdater_;

  Float (*sampler_)(const Config& cfg, std::vector<Edge>* edges,
                    unsigned int* seed);

  uint32_t stepCount_;
  uint64_t time_;
  uint64_t samplingTime_;
  Sample samples_[2];
  std::future<Float> futures_[2];
  int phase_;
};

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
