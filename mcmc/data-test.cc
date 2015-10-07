#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>

#include "mcmc/data.h"

using namespace mcmc;

std::vector<Edge> GenerateRandom(uint32_t node_max) {
  std::vector<Edge> edges(1024);
  std::default_random_engine generator;
  std::uniform_int_distribution<Vertex> distribution(0, node_max);
  for (auto &e : edges) {
    Vertex x = distribution(generator) % node_max;
    Vertex y = distribution(generator) % node_max;
    Vertex u = std::min(x, y);
    Vertex v = std::max(x, y);
    e = MakeEdge(u, v);
  }
  std::sort(edges.begin(), edges.end());
  auto end = std::unique(edges.begin(), edges.end());
  edges.resize(end - edges.begin());
  return edges;
}

TEST(DataTest, RandomEdge) {
  const uint32_t N = 1024;
  std::vector<Edge> edges = GenerateRandom(N);
  Graph g(N, edges);
  for (uint32_t i = 0; i < N; ++i) {
    Edge e = g.GetRandomEdge();
    Vertex u, v;
    std::tie(u, v) = Vertices(e);
    auto& u_adj = g.NeighborsOf(u);
    ASSERT_NE(std::find(u_adj.begin(), u_adj.end(), v), u_adj.end());
    auto& v_adj = g.NeighborsOf(v);
    ASSERT_NE(std::find(v_adj.begin(), v_adj.end(), u), v_adj.end());
  }
  for (uint32_t i = 0; i < N; ++i) {
    Vertex u, v;
    Edge e;
    do {
      u = rand() % N;
      v = rand() % N;
      e = MakeEdge(u, v);
    } while (std::find(edges.begin(), edges.end(), e) != edges.end());
    auto& u_adj = g.NeighborsOf(u);
    ASSERT_EQ(std::find(u_adj.begin(), u_adj.end(), v), u_adj.end());
    auto& v_adj = g.NeighborsOf(v);
    ASSERT_EQ(std::find(v_adj.begin(), v_adj.end(), u), v_adj.end());
  }
}
