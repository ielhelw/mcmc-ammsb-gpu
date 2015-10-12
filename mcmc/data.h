#ifndef __MCMC_DATA_H__
#define __MCMC_DATA_H__

#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "mcmc/cuckoo.h"

namespace mcmc {

// Drop in mcmc::cuckoo set implementation
using namespace mcmc::cuckoo;

class Graph {
 public:
  Graph(uint32_t num_nodes, const std::vector<Edge>& unique_edges);

  Edge GetRandomEdge() const;

  inline const std::vector<Vertex>& NeighborsOf(Vertex u) const {
    return adjacency_[u];
  }

  inline const std::vector<Edge>& UniqueEdges() const { return unique_edges_; }

 private:
  uint32_t num_nodes_;
  std::vector<Edge> unique_edges_;
  std::vector<std::vector<Vertex>> adjacency_;
};

bool GetUniqueEdgesFromFile(const std::string& filename,
                            uint64_t* count_vertices, std::vector<Edge>* vals);

bool GenerateSetsFromEdges(const std::vector<Edge>& vals, double heldout_ratio,
                           std::vector<Edge>* training_edges,
                           std::vector<Edge>* heldout_edges,
                           std::unique_ptr<Set>* training,
                           std::unique_ptr<Set>* heldout);

bool GenerateSetsFromFile(const std::string& filename, double heldout_ratio,
                          uint64_t* count_vertices,
                          std::vector<Edge>* training_edges,
                          std::vector<Edge>* heldout_edges,
                          std::unique_ptr<Set>* training,
                          std::unique_ptr<Set>* heldout);

}  // namespace mcmc

#endif  // __MCMC_DATA_H__
