#ifndef __MCMC_SAMPLE_H__
#define __MCMC_SAMPLE_H__

#include <boost/program_options.hpp>
#include <ostream>

#include "mcmc/types.h"

namespace mcmc {

struct Config;

struct Sample {
  std::vector<Edge> edges;
  clcuda::Buffer<Edge> dev_edges;
  std::vector<Vertex> nodes_vec;
  clcuda::Buffer<Vertex> dev_nodes;
  std::vector<Vertex> neighbors_vec;
  clcuda::Buffer<Vertex> dev_neighbors;
  std::vector<unsigned int> seeds;

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
