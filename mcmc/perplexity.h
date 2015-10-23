#ifndef __MCMC_PERPLEXITY_H__
#define __MCMC_PERPLEXITY_H__

#include "mcmc/types.h.inc"

#ifndef MCMC_USE_CL
#include <thrust/device_vector.h>
#endif

#include "mcmc/config.h"
#include "mcmc/partitioned-alloc.h"

#ifdef MCMC_USE_CL
#include <boost/compute/command_queue.hpp>
#include <boost/compute/container/vector.hpp>
#endif

namespace mcmc {

class PerplexityCalculatorBase {
 public:
  enum Mode {
    EDGE_PER_THREAD,
    EDGE_PER_WORKGROUP
  };

  PerplexityCalculatorBase(Mode mode, const Config& cfg, clcuda::Queue queue,
                       clcuda::Buffer<Float>& beta,
                       RowPartitionedMatrix<Float>* pi,
                       clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
                       const std::vector<std::string>& compileFlags,
                       const std::string& baseFuncs);

  Float operator()();

  uint64_t LastInvocationTime() const;

 protected:
  virtual void AccumulateVectors() = 0;

  clcuda::Queue queue_;
  clcuda::Event event_;

  clcuda::Buffer<Float>& beta_;      // [K]
  RowPartitionedMatrix<Float>* pi_;  // [N,K]
  clcuda::Buffer<Edge>& edges_;
  OpenClSet* edgeSet_;

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

#ifdef MCMC_USE_CL
class PerplexityCalculatorCl : public PerplexityCalculatorBase {
 public:
  PerplexityCalculatorCl(Mode mode, const Config& cfg, clcuda::Queue queue,
                         clcuda::Buffer<Float>& beta,
                         RowPartitionedMatrix<Float>* pi,
                         clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
                         const std::vector<std::string>& compileFlags,
                         const std::string& baseFuncs);
 private:
  void AccumulateVectors() override;

  boost::compute::command_queue compute_queue_;
  boost::compute::vector<Float> ppx_per_edge_;
  boost::compute::vector<Float> ppx_per_edge_link_likelihood_;
  boost::compute::vector<Float> ppx_per_edge_non_link_likelihood_;
  boost::compute::vector<uint32_t> ppx_per_edge_link_count_;
  boost::compute::vector<uint32_t> ppx_per_edge_non_link_count_;
};
#else
class PerplexityCalculatorCu : public PerplexityCalculatorBase {
 public:
  PerplexityCalculatorCu(Mode mode, const Config& cfg, clcuda::Queue queue,
                         clcuda::Buffer<Float>& beta,
                         RowPartitionedMatrix<Float>* pi,
                         clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
                         const std::vector<std::string>& compileFlags,
                         const std::string& baseFuncs);
 private:
  void AccumulateVectors();

  thrust::device_vector<Float> ppx_per_edge_;
  thrust::device_vector<Float> ppx_per_edge_link_likelihood_;
  thrust::device_vector<Float> ppx_per_edge_non_link_likelihood_;
  thrust::device_vector<uint32_t> ppx_per_edge_link_count_;
  thrust::device_vector<uint32_t> ppx_per_edge_non_link_count_;
};
#endif

#ifdef MCMC_USE_CL
typedef PerplexityCalculatorCl PerplexityCalculator;
#else
typedef PerplexityCalculatorCu PerplexityCalculator;
#endif

}  // namespace mcmc

#endif  //  __MCMC_PERPLEXITY_H__
