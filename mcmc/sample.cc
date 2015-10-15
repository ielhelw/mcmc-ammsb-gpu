#include "mcmc/sample.h"

#include <unordered_set>
#include <queue>

#include "mcmc/config.h"

namespace mcmc {

Sample::Sample(const Config& cfg, compute::command_queue queue)
    : dev_edges(cfg.mini_batch_size, queue.get_context()),
      dev_nodes(2 * cfg.mini_batch_size, queue.get_context()),
      dev_neighbors(2 * cfg.mini_batch_size * cfg.num_node_sample,
                    queue.get_context()),
      seeds(2 * cfg.mini_batch_size) {
  std::generate(seeds.begin(), seeds.end(), rand);
}

std::istream& operator>>(std::istream& in, SampleStrategy& strategy) {
  std::string token;
  in >> token;
  if (boost::iequals(token, "NodeLink")) {
    strategy = NodeLink;
  } else if (boost::iequals(token, "NodeNonLink")) {
    strategy = NodeNonLink;
  } else if (boost::iequals(token, "NodeStratified")) {
    strategy = NodeStratified;
  } else {
    throw boost::program_options::validation_error(boost::program_options::validation_error::invalid_option_value, "Invalid SampleStrategy");
  }
  return in;
}

void sampleNodeStratified(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed) {
  std::unordered_set<Vertex> Us;
  std::queue<Vertex> q;
  std::unordered_set<Edge> set;
  while (set.size() < cfg.mini_batch_size) {
    if (q.empty()) {
      Vertex u;
      do {
        u = rand_r(seed) % cfg.N;
      } while (Us.find(u) != Us.end());
      q.push(u);
    }
    Vertex u = q.front();
    q.pop();
    if (Us.insert(u).second) {
      auto& neighbors = cfg.trainingGraph->NeighborsOf(u);
      for (auto v : neighbors) {
        if (set.size() < cfg.mini_batch_size) {
          q.push(v);
          set.insert(MakeEdge(std::min(u, v), std::max(u, v)));
        } else {
          break;
        }
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
}

// 1- randomly select u
// 2- select all training edges (u, v) 
// 3 repeat until mini-batch is full
void sampleNodeLink(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed) {
  std::unordered_set<Vertex> Us;
  std::unordered_set<Edge> set;
  while (set.size() < cfg.mini_batch_size) {
    Vertex u = rand_r(seed) % cfg.N;
    if (Us.insert(u).second) {
      auto& neighbors = cfg.trainingGraph->NeighborsOf(u);
      for (auto& v : neighbors) {
        if (set.size() < cfg.mini_batch_size) {
          Edge e = MakeEdge(std::min(u, v), std::max(u, v));
          set.insert(e);
        } else {
          break;
        }
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
}

// 1- randomly select u
// 2- ramdomly select multiple Vs
// 3- make sure (u, v) not in training or heldout
// 4- repeat until mini-batch is full
void sampleNodeNonLink(const Config& cfg, std::vector<Edge>* edges, unsigned int* seed) {
  std::unordered_set<Edge> set;
  std::unordered_set<Vertex> Vs;
  Vertex u = rand_r(seed) % cfg.N;
  Vertex v;
  Edge e;
  while (set.size() < cfg.mini_batch_size) {
    do {
      v = rand_r(seed) % cfg.N;
      e = MakeEdge(std::min(u, v), std::max(u, v));
    } while (Vs.find(v) != Vs.end() || cfg.heldout->Has(e) || cfg.training->Has(e));
    set.insert(e);
  }
  edges->insert(edges->begin(), set.begin(), set.end());
}

}  // namespace mcmc
