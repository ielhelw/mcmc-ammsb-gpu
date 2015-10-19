#ifndef __MCMC_LEARNER_H__
#define __MCMC_LEARNER_H__

#include <boost/compute/system.hpp>
#include <boost/compute/container/array.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/program_options.hpp>
#include <ostream>

#include "mcmc/algorithm/normalize.h"
#include "mcmc/config.h"
#include "mcmc/data.h"
#include "mcmc/perplexity.h"
#include "mcmc/phi.h"
#include "mcmc/beta.h"

namespace mcmc {

class Learner {
 public:

  Learner(const Config& cfg, compute::command_queue queue);

  Float calculate_perplexity_heldout(uint32_t step_count);

  Float sampleMiniBatch(std::vector<Edge>* edges, unsigned int* seed);

  void extractNodesFromMiniBatch(const std::vector<Edge>& edges,
                                 std::vector<Vertex>* nodes);

  void sampleNeighbors(const std::vector<Vertex>& nodes,
                       std::vector<Vertex>* neighbors,
                       std::vector<unsigned int>* seeds);

  Float DoSample(Sample* sample);

  void run(uint32_t max_iters);

  static const std::string& GetBaseFuncs();

 private:
  const Config& cfg_;

  compute::command_queue queue_;
  compute::vector<Float> beta_;   // [K]
  compute::vector<Float> theta_;  // [K]

  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory_;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;   // [N,K]
  compute::vector<Float> phi_;  // [N]

  std::shared_ptr<OpenClSetFactory> setFactory_;

  std::unique_ptr<OpenClSet> trainingSet_;
  std::unique_ptr<OpenClSet> heldoutSet_;
  compute::vector<Edge> trainingEdges_;
  compute::vector<Edge> heldoutEdges_;

  std::string compileFlags_;

  PerplexityCalculator heldoutPerplexity_;
  PhiUpdater phiUpdater_;
  BetaUpdater betaUpdater_;
  
  Float (*sampler_)(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed);
};

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
