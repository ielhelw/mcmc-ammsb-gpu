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

  // Generate base->size() elements in base. Normalized the first num_rows*K in
  // norm.
  template <class Generator>
  static void GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen, compute::vector<Float>* base,
                                   compute::vector<Float>* norm, uint32_t cols);

  template <class Generator>
  static void GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen,
                                   compute::vector<Float>& sum,
                                   RowPartitionedMatrix<Float>* norm);

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

template <class Generator>
void Learner::GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen, compute::vector<Float>* base,
                                   compute::vector<Float>* norm,
                                   uint32_t cols) {
  std::vector<Float> host_base(base->size());
  std::generate(host_base.begin(), host_base.end(), *gen);
  compute::copy(host_base.begin(), host_base.end(), base->begin(), *queue);
  compute::copy(base->begin(), base->end(), norm->begin(), *queue);
  mcmc::algorithm::Normalizer<Float>(*queue, norm, cols, 1)();
}

template <class Generator>
void Learner::GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen,
                                   compute::vector<Float>& sum,
                                   RowPartitionedMatrix<Float>* norm) {
  for (uint32_t i = 0; i < norm->Blocks().size(); ++i) {
    std::vector<Float> host_base(norm->Blocks()[i].size());
    std::generate(host_base.begin(), host_base.end(), *gen);
    compute::copy(host_base.begin(), host_base.end(), norm->Blocks()[i].begin(),
                  *queue);
  }
  mcmc::algorithm::PartitionedNormalizer<Float>(*queue, norm, sum, 32)();
}

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
