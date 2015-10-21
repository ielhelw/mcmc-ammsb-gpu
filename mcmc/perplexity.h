#ifndef __MCMC_PERPLEXITY_H__
#define __MCMC_PERPLEXITY_H__

#include "mcmc/config.h"

#include "mcmc/partitioned-alloc.h"

namespace mcmc {

class PerplexityCalculator {
 public:
  enum Mode { EDGE_PER_THREAD, EDGE_PER_WORKGROUP };

  PerplexityCalculator(Mode mode, const Config& cfg, clcuda::Queue queue,
                       clcuda::Buffer<Float>& beta,
                       RowPartitionedMatrix<Float>* pi,
                       clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
                       const std::string& compileFlags,
                       const std::string& baseFuncs);

  Float operator()();

  uint64_t LastInvocationTime() const;

 private:
  clcuda::Queue queue_;
  clcuda::Event event_;

  clcuda::Buffer<Float>& beta_;      // [K]
  RowPartitionedMatrix<Float>* pi_;  // [N,K]
  clcuda::Buffer<Edge>& edges_;
  OpenClSet* edgeSet_;

  clcuda::Buffer<Float> ppx_per_edge_;
  clcuda::Buffer<Float> ppx_per_edge_link_likelihood_;
  clcuda::Buffer<Float> ppx_per_edge_non_link_likelihood_;
  clcuda::Buffer<uint32_t> ppx_per_edge_link_count_;
  clcuda::Buffer<uint32_t> ppx_per_edge_non_link_count_;

  std::vector<uint32_t> link_count_;
  std::vector<uint32_t> non_link_count_;
  std::vector<Float> link_likelihood_;
  std::vector<Float> non_link_likelihood_;

  std::unique_ptr<clcuda::Buffer<Float>> scratch_;
  std::unique_ptr<clcuda::Program> prog_;
  std::unique_ptr<clcuda::Kernel> kernel_;

  uint32_t count_calls_;
  uint32_t global_, local_;
};

}  // namespace mcmc

#endif  //  __MCMC_PERPLEXITY_H__
