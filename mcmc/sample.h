#ifndef __MCMC_SAMPLE_H__
#define __MCMC_SAMPLE_H__

#include <boost/compute/system.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/program_options.hpp>
#include <ostream>

#include "mcmc/types.h"

namespace mcmc {

struct Config;

struct Sample {
  std::vector<Edge> edges;
  compute::vector<Edge> dev_edges;
  std::vector<Vertex> nodes_vec;
  compute::vector<Vertex> dev_nodes;
  std::vector<Vertex> neighbors_vec;
  compute::vector<Vertex> dev_neighbors;
  std::vector<unsigned int> seeds;

  Sample(const Config& cfg, compute::command_queue queue);
};

enum SampleStrategy {
  NodeLink,
  NodeNonLink,
  NodeStratified,
};

std::istream& operator>>(std::istream& in, SampleStrategy& strategy);

void sampleNodeStratified(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed);

void sampleNodeLink(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed);

void sampleNodeNonLink(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed);

}  // namespace mcmc

#endif  // __MCMC_SAMPLE_H__
