#include "mcmc/data.h"

#include <cmath>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <glog/logging.h>

namespace mcmc {

Graph::Graph(uint64_t num_nodes, const std::vector<Edge>& unique_edges)
    : num_nodes_(num_nodes),
      unique_edges_(unique_edges),
      adjacency_(num_nodes),
      max_fan_out_(0) {
  for (auto e : unique_edges_) {
    Vertex u, v;
    std::tie(u, v) = Vertices(e);
    adjacency_[u].push_back(v);
    adjacency_[v].push_back(u);
    max_fan_out_ = std::max(adjacency_[u].size(), adjacency_[v].size());
  }
}

Edge Graph::GetRandomEdge() const {
  Vertex u;
  do {
    u = rand() % num_nodes_;
  } while (adjacency_[u].empty());
  Vertex v = *(adjacency_[u].begin() + rand() % adjacency_[u].size());
  return MakeEdge(u, v);
}

bool GetUniqueEdgesFromFile(const std::string& filename,
                            uint64_t* count_vertices, std::vector<Edge>* vals) {
  std::set<Vertex> unique_vertices;
  std::vector<Edge> edges;
  std::ifstream in(filename);
  std::string line;
  // skip first 4 lines
  for (int i = 0; i < 4; ++i) std::getline(in, line);
  do {
    uint64_t a, b, x, y;
    in >> a >> b;
    if (!in.eof()) {
      x = std::min(a, b);
      y = std::max(a, b);
      edges.push_back(MakeEdge(x, y));
      unique_vertices.insert(x);
      unique_vertices.insert(y);
    }
  } while (in.good());
  if (in.bad()) {
    LOG(ERROR) << "Error reading file " << filename;
    return false;
  }
  // rename vertices in range [0, N-1]
  std::unordered_map<Vertex, Vertex> map;
  Vertex i = 0;
  for (auto v : unique_vertices) {
    map[v] = i++;
  }
  *count_vertices = map.size();
  for (auto e : edges) {
    Vertex u, v;
    std::tie(u, v) = Vertices(e);
    vals->push_back(MakeEdge(map[u], map[v]));
  }
  std::sort(vals->begin(), vals->end());
  // squeeze out duplicates
  auto end = std::unique(vals->begin(), vals->end());
  vals->resize(end - vals->begin());
  // shuffle again
  std::random_shuffle(vals->begin(), vals->end());
  return true;
}

bool GenerateSetsFromEdges(uint64_t N, const std::vector<Edge>& vals,
                           double heldout_ratio,
                           std::vector<Edge>* training_edges,
                           std::vector<Edge>* heldout_edges,
                           std::unique_ptr<Set>* training,
                           std::unique_ptr<Set>* heldout) {
  size_t training_len =
      static_cast<size_t>(std::ceil((1 - heldout_ratio / 2) * vals.size()));
  size_t heldout_len = vals.size() - training_len;
  if (heldout_len > 0) {
    heldout->reset(new Set(heldout_len));
    if (!(*heldout)->SetContents(vals.begin(), vals.begin() + heldout_len)) {
      LOG(ERROR) << "Failed to insert into heldout set";
      heldout->reset();
      return false;
    }
    heldout_edges->insert(heldout_edges->end(), vals.begin(),
                          vals.begin() + heldout_len);
  }
  training->reset(new Set(training_len));
  if (!(*training)->SetContents(vals.begin() + heldout_len, vals.end())) {
    LOG(ERROR) << "Failed to insert into training set";
    training->reset();
    if (heldout_len > 0) {
      heldout->reset();
    }
    return false;
  }
  training_edges->insert(training_edges->end(), vals.begin() + heldout_len,
                         vals.end());
  if (heldout_len > 0) {
    std::unordered_set<Edge> fake_edges;
    for (uint32_t i = 0; i < heldout_len; ++i) {
      Vertex u, v;
      Edge e;
      do {
        u = rand() % N;
        do {
          v = rand() % N;
        } while (u == v);  // no self links
        e = MakeEdge(std::min(u, v), std::max(u, v));
      } while (fake_edges.find(e) != fake_edges.end() || (*heldout)->Has(e) ||
               (*training)->Has(e));  // skip edge
      fake_edges.insert(e);
      heldout_edges->push_back(e);
    }
  }
  return true;
}

bool GenerateSetsFromFile(const std::string& filename, double heldout_ratio,
                          uint64_t* count_vertices,
                          std::vector<Edge>* training_edges,
                          std::vector<Edge>* heldout_edges,
                          std::unique_ptr<Set>* training,
                          std::unique_ptr<Set>* heldout) {
  LOG(INFO) << "Going to generate sets from " << filename
            << " with held-out ratio " << heldout_ratio;
  std::vector<Edge> vals;
  if (!GetUniqueEdgesFromFile(filename, count_vertices, &vals)) return false;
  if (!GenerateSetsFromEdges(*count_vertices, vals, heldout_ratio,
                             training_edges, heldout_edges, training, heldout))
    return false;
  return true;
}

}  // namespace mcmc
