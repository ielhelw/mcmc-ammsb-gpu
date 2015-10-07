#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/test.h"
#include "mcmc/perplexity.h"
#include "mcmc/learner.h"

namespace mcmc {
namespace test {

std::vector<Edge> GenerateRandomEdges(uint32_t num_edges, Vertex max_id) {
  std::vector<Edge> edges(num_edges);
  std::default_random_engine generator;
  std::uniform_int_distribution<Vertex> distribution(0, max_id);
  for (auto& e : edges) {
    Vertex u = distribution(generator);
    Vertex v = distribution(generator);
    e = MakeEdge(std::min(u, v), std::max(u, v));
  }
  std::sort(edges.begin(), edges.end());
  auto end = std::unique(edges.begin(), edges.end());
  edges.resize(end - edges.begin());
  return edges;
}

class WgPerplexityTest : public ContextTest,
                         public ::testing::WithParamInterface<uint32_t> {};

TEST_P(WgPerplexityTest, Equal) {
  const uint32_t N = 1024;
  const uint32_t K = 1024;
  const uint32_t num_tries = 10;
  std::vector<Edge> edges = GenerateRandomEdges(10 * 1024, N);
  compute::vector<Edge> dev_edges(edges.begin(), edges.end(), queue_);
  Set set(edges.size());
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    ASSERT_TRUE(set.Insert(*it));
  }
  auto factory = OpenClSetFactory::New(queue_);
  std::unique_ptr<OpenClSet> dev_set(factory->CreateSet(set.Serialize()));
  std::mt19937 mt19937;
  std::normal_distribution<float> norm_dist(0, 1);
  auto gen = std::bind(norm_dist, mt19937);
  std::vector<float> pi(N * K);
  std::generate(pi.begin(), pi.end(), gen);
  std::vector<float> beta(K);
  std::generate(beta.begin(), beta.end(), gen);
  compute::vector<float> dev_pi(pi.begin(), pi.end(), queue_);
  compute::vector<float> dev_beta(beta.begin(), beta.end(), queue_);
  Config cfg;
  cfg.K = K;
  cfg.ppx_wg_size = GetParam();
  mcmc::PerplexityCalculator ppxSimple(mcmc::PerplexityCalculator::EDGE_PER_THREAD,
                                 cfg, queue_, dev_beta, dev_pi, dev_edges,
                                 dev_set.get(), MakeCompileFlags(cfg),
                                 Learner::kSourceBaseFuncs);
  auto ppx1 = ppxSimple();
  double ppx1_time = 0;
  for (uint32_t i = 0; i < num_tries; ++i) {
    ASSERT_FLOAT_EQ(ppx1, ppxSimple());
    ppx1_time += ppxSimple.LastInvocationTime();
  }
  mcmc::PerplexityCalculator ppxWg(mcmc::PerplexityCalculator::EDGE_PER_WORKGROUP,
                                 cfg, queue_, dev_beta, dev_pi, dev_edges,
                                 dev_set.get(), MakeCompileFlags(cfg),
                                 Learner::kSourceBaseFuncs);
  auto ppx2 = ppxWg();
  double ppx2_time = 0;
  ASSERT_FLOAT_EQ(ppx1, ppx2);
  for (uint32_t i = 0; i < num_tries; ++i) {
    ASSERT_FLOAT_EQ(ppx2, ppxWg());
    ppx2_time += ppxWg.LastInvocationTime();
  }
  LOG(INFO) << "K=" << K << ", WG=" << cfg.ppx_wg_size
    << ", EDGE_PER_THREAD=" << ppx1_time / num_tries
    << ", EDGE_PER_WG=" << ppx2_time / num_tries
    << "(" << (ppx1_time / ppx2_time) << ")";
}
INSTANTIATE_TEST_CASE_P(WorkGroups, WgPerplexityTest,
                        ::testing::ValuesIn(std::vector<uint32_t>(
                            {2, 4, 16, 32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
