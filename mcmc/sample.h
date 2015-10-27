#ifndef __MCMC_SAMPLE_H__
#define __MCMC_SAMPLE_H__

#include <ostream>

#include "mcmc/types.h"
#include "mcmc/random.h"

namespace mcmc {

std::string GetNeighborSamplerSource();

struct Config;

class NeighborSampler {
 public:
  NeighborSampler(const Config& cfg, clcuda::Queue queue);

  void operator()(uint32_t num_samples, clcuda::Buffer<Vertex>* nodes);

  clcuda::Buffer<Vertex>& GetHash() { return hash_; }

  clcuda::Buffer<Vertex>& GetData() { return data_; }

  uint32_t HashCapacityPerSample();

  uint32_t DataSizePerSample();

 private:
  const Config& cfg_;
  uint32_t capacity_;
  uint32_t local_;
  clcuda::Queue queue_;
  clcuda::Buffer<Vertex> hash_;
  clcuda::Buffer<Vertex> data_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> kernel_;
  std::shared_ptr<random::OpenClRandomFactory> randFactory_;
  std::unique_ptr<random::OpenClRandom> rand_;
};

struct Sample {
  clcuda::Queue queue;
  std::vector<Edge> edges;
  clcuda::Buffer<Edge> dev_edges;
  std::vector<Vertex> nodes_vec;
  clcuda::Buffer<Vertex> dev_nodes;
  unsigned int seed;
  NeighborSampler neighbor_sampler;

  Sample(const Config& cfg, clcuda::Queue queue);
};

enum SampleStrategy {
  Node,
  NodeLink,
  NodeNonLink,
  BFLink,
  BFNonLink,
  BF,
};

std::string to_string(SampleStrategy s);

std::istream& operator>>(std::istream& in, SampleStrategy& strategy);

Float sampleBreadthFirstLink(const Config& cfg, std::vector<Edge>* edges,
                             unsigned int* seed);

Float sampleBreadthFirstNonLink(const Config& cfg, std::vector<Edge>* edges,
                                unsigned int* seed);

Float sampleBreadthFirst(const Config& cfg, std::vector<Edge>* edges,
                         unsigned int* seed);

Float sampleNodeLink(const Config& cfg, std::vector<Edge>* edges,
                     unsigned int* seed);

Float sampleNodeNonLink(const Config& cfg, std::vector<Edge>* edges,
                        unsigned int* seed);

Float sampleNode(const Config& cfg, std::vector<Edge>* edges,
                 unsigned int* seed);

}  // namespace mcmc

#endif  // __MCMC_SAMPLE_H__
