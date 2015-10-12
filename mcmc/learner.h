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

namespace mcmc {

class Learner {
 public:
  Learner(const Config& cfg, compute::command_queue queue);

  Float calculate_perplexity_heldout(uint32_t step_count);

  void run();

  // Generate base->size() elements in base. Normalized the first num_rows*K in
  // norm.
  template <class Generator>
  static void GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen, compute::vector<Float>* base,
                                   compute::vector<Float>* norm,
                                   uint32_t num_rows, uint32_t K);

  template <class Generator>
  static void GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen,
                                   RowPartitionedMatrix<Float>* base,
                                   RowPartitionedMatrix<Float>* norm);

  static const std::string& GetBaseFuncs();

 private:
  const Config& cfg_;

  compute::command_queue queue_;
  compute::vector<Float> beta_;   // [K]
  compute::vector<Float> theta_;  // [K]

  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory_;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;   // [N,K]
  std::unique_ptr<RowPartitionedMatrix<Float>> phi_;  // [N,K]

  compute::buffer scratch_;

  std::shared_ptr<OpenClSetFactory> setFactory_;
  std::unique_ptr<OpenClSet> trainingSet_;
  std::unique_ptr<OpenClSet> heldoutSet_;
  compute::vector<Edge> trainingEdges_;
  compute::vector<Edge> heldoutEdges_;

  std::string compileFlags_;

  PerplexityCalculator heldoutPerplexity_;
  PhiUpdater phiUpdater_;
};

template <class Generator>
void Learner::GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen, compute::vector<Float>* base,
                                   compute::vector<Float>* norm,
                                   uint32_t num_rows, uint32_t K) {
  std::vector<Float> host_base(base->size());
  std::generate(host_base.begin(), host_base.end(), *gen);
  compute::copy(host_base.begin(), host_base.end(), base->begin(), *queue);
  compute::copy(base->begin(), base->begin() + K * num_rows, norm->begin(),
                *queue);
  mcmc::algorithm::Normalizer<Float>(*queue, norm, K, 32)();
}

template <class Generator>
void Learner::GenerateAndNormalize(compute::command_queue* queue,
                                   Generator* gen,
                                   RowPartitionedMatrix<Float>* base,
                                   RowPartitionedMatrix<Float>* norm) {
  for (uint32_t i = 0; i < base->Blocks().size(); ++i) {
    std::vector<Float> host_base(base->Blocks()[0].size());
    std::generate(host_base.begin(), host_base.end(), *gen);
    compute::copy(host_base.begin(), host_base.end(), base->Blocks()[i].begin(),
                  *queue);
    compute::copy(base->Blocks()[i].begin(), base->Blocks()[i].end(),
                  norm->Blocks()[i].begin(), *queue);
  }
  mcmc::algorithm::PartitionedNormalizer<Float>(*queue, norm, 32)();
}

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
