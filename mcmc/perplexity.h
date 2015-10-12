#ifndef __MCMC_PERPLEXITY_H__
#define __MCMC_PERPLEXITY_H__

#include "mcmc/config.h"

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>

#include "mcmc/partitioned-alloc.h"

namespace mcmc {

class PerplexityCalculator {
 public:
  enum Mode { EDGE_PER_THREAD, EDGE_PER_WORKGROUP };

  PerplexityCalculator(Mode mode, const Config& cfg,
                       compute::command_queue queue,
                       compute::vector<Float>& beta, RowPartitionedMatrix<Float>* pi,
                       compute::vector<Edge>& edges, OpenClSet* edgeSet,
                       const std::string& compileFlags,
                       const std::string& baseFuncs);

  Float operator()();

  uint64_t LastInvocationTime() const;

 private:
  compute::command_queue queue_;
  compute::event event_;

  compute::vector<Float>& beta_;  // [K]
  RowPartitionedMatrix<Float>* pi_;    // [N,K]
  compute::vector<Edge>& edges_;
  OpenClSet* edgeSet_;

  compute::vector<Float> ppx_per_edge_;
  compute::vector<Float> ppx_per_edge_link_likelihood_;
  compute::vector<Float> ppx_per_edge_non_link_likelihood_;
  compute::vector<compute::uint_> ppx_per_edge_link_count_;
  compute::vector<compute::uint_> ppx_per_edge_non_link_count_;

  std::vector<compute::uint_> link_count_;
  std::vector<compute::uint_> non_link_count_;
  std::vector<Float> link_likelihood_;
  std::vector<Float> non_link_likelihood_;
  
  compute::vector<Float> scratch_;
  compute::program prog_;
  compute::kernel kernel_;

  uint32_t count_calls_;
  uint32_t global_, local_;
};

}  // namespace mcmc

#endif  //  __MCMC_PERPLEXITY_H__
