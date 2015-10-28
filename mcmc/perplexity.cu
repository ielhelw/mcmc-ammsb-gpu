#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <thrust/reduce.h>
#pragma GCC diagnostic pop

#include "mcmc/perplexity.h"

namespace mcmc {

PerplexityCalculatorCu::PerplexityCalculatorCu(Mode mode, const Config& cfg, clcuda::Queue queue,
                         clcuda::Buffer<Float>& beta,
                         RowPartitionedMatrix<Float>* pi,
                         clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
                         const std::vector<std::string>& compileFlags,
                         const std::string& baseFuncs) : PerplexityCalculatorBase(mode, cfg, queue, beta, pi, edges, edgeSet, compileFlags, baseFuncs),
  ppx_per_edge_(edges_.GetSize() / sizeof(Edge), 0),
  ppx_per_edge_link_likelihood_(edges.GetSize() / sizeof(Edge), 0),
  ppx_per_edge_non_link_likelihood_(edges.GetSize() / sizeof(Edge), 0),
  ppx_per_edge_link_count_(edges.GetSize() / sizeof(Edge), 0),
  ppx_per_edge_non_link_count_(edges.GetSize() / sizeof(Edge), 0) {
  kernel_->SetArgument(5, ppx_per_edge_.data());
  kernel_->SetArgument(6, ppx_per_edge_link_likelihood_.data());
  kernel_->SetArgument(7, ppx_per_edge_non_link_likelihood_.data());
  kernel_->SetArgument(8, ppx_per_edge_link_count_.data());
  kernel_->SetArgument(9, ppx_per_edge_non_link_count_.data());
}

void PerplexityCalculator::AccumulateVectors() {
  link_count_[0] = thrust::reduce(ppx_per_edge_link_count_.begin(),
                                  ppx_per_edge_link_count_.end());
  non_link_count_[0] = thrust::reduce(ppx_per_edge_non_link_count_.begin(),
                                      ppx_per_edge_non_link_count_.end());
  link_likelihood_[0] = thrust::reduce(ppx_per_edge_link_likelihood_.begin(),
                                       ppx_per_edge_link_likelihood_.end());
  non_link_likelihood_[0] =
      thrust::reduce(ppx_per_edge_non_link_likelihood_.begin(),
                     ppx_per_edge_non_link_likelihood_.end());
}

}  // namespace mcmc
