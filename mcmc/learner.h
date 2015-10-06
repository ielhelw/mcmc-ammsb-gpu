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

namespace mcmc {

class Learner {
 public:

  Learner(const Config& cfg, compute::command_queue queue);

  Float calculate_perplexity_heldout(uint32_t step_count);

  void run();

 private:
  const Config& cfg_;

  compute::command_queue queue_;
  compute::vector<Float> beta_;   // [K]
  compute::vector<Float> theta_;  // [K]
  compute::vector<Float> pi_;     // [N,K]
  compute::vector<Float> phi_;    // [N,K]

  compute::buffer scratch_;
  
  std::shared_ptr<OpenClSetFactory> setFactory_;
  std::unique_ptr<OpenClSet> trainingSet_;
  std::unique_ptr<OpenClSet> heldoutSet_;
  compute::vector<Edge> trainingEdges_;
  compute::vector<Edge> heldoutEdges_;

  std::string compileFlags_;
  
  PerplexityCalculator heldoutPerplexity_;
};

std::ostream& operator<<(std::ostream& out, const Config& cfg);

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
