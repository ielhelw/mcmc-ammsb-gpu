#ifndef __MCMC_SAMPLE_H__
#define __MCMC_SAMPLE_H__

#include <ostream>

#include "mcmc/types.h"
#include "mcmc/random.h"
#include "mcmc/serialize.h"

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

  inline bool Serialize(std::ostream* out) {
    return rand_->Serialize(out) && ::mcmc::Serialize(out, &data_, &queue_);
  }

  inline bool Parse(std::istream* in) {
    return rand_->Parse(in) && ::mcmc::Parse(in, &data_, &queue_);
  }

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

  inline bool Serialize(std::ostream* out) {
    SampleStorage storage;
    storage.mutable_edges()->resize(edges.size() * sizeof(Edge));
    memcpy((void*)storage.mutable_edges()->data(), edges.data(),
           edges.size() * sizeof(Edge));
    storage.mutable_nodes_vec()->resize(nodes_vec.size() * sizeof(Vertex));
    memcpy((void*)storage.mutable_nodes_vec()->data(), nodes_vec.data(),
           nodes_vec.size() * sizeof(Vertex));
    storage.set_seed(seed);
    return ::mcmc::SerializeMessage(out, storage) &&
           ::mcmc::Serialize(out, &dev_edges, &queue) &&
           ::mcmc::Serialize(out, &dev_nodes, &queue) &&
           neighbor_sampler.Serialize(out);
  }

  inline bool Parse(std::istream* in) {
    SampleStorage storage;
    if (::mcmc::ParseMessage(in, &storage) &&
        ::mcmc::Parse(in, &dev_edges, &queue) &&
        ::mcmc::Parse(in, &dev_nodes, &queue) && neighbor_sampler.Parse(in)) {
      edges.resize(storage.edges().size() / sizeof(Edge));
      memcpy(edges.data(), storage.edges().data(), edges.size() * sizeof(Edge));
      nodes_vec.resize(storage.nodes_vec().size() / sizeof(Vertex));
      memcpy(nodes_vec.data(), storage.nodes_vec().data(),
             nodes_vec.size() * sizeof(Vertex));
      seed = storage.seed();
      return true;
    }
    return false;
  }
};

enum SampleStrategy {
  Node,
  NodeLink,
  NodeNonLink,
  BFLink,
  BFNonLink,
  BF,
};

std::string to_string(const SampleStrategy& s);

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
