#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>
#include <chrono>

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
                         public ::testing::WithParamInterface<uint32_t> {
 protected:
  WgPerplexityTest(uint32_t K = 1024, uint32_t N = 1024)
    : N_(N), K_(K), num_tries_(10) {}

  void SetUp() override {
    ContextTest::SetUp();
    std::vector<Edge> edges = GenerateRandomEdges(1024, N_);
    dev_edges_ = compute::vector<Edge>(edges.begin(), edges.end(), queue_);
    Set set(edges.size());
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      ASSERT_TRUE(set.Insert(*it));
    }
    factory_ = OpenClSetFactory::New(queue_);
    dev_set_.reset(factory_->CreateSet(set.Serialize()));
    std::mt19937 mt19937;
    std::normal_distribution<Float> norm_dist(0, 1);
    auto gen = std::bind(norm_dist, mt19937);
    std::vector<Float> pi(N_ * K_);
    std::generate(pi.begin(), pi.end(), gen);
    std::vector<Float> beta(K_);
    std::generate(beta.begin(), beta.end(), gen);
    dev_pi_ = compute::vector<Float>(pi.begin(), pi.end(), queue_);
    dev_beta_ = compute::vector<Float>(beta.begin(), beta.end(), queue_);
    cfg_.K = K_;
  }
  
  void TearDown() override {
    dev_edges_ = compute::vector<Edge>();
    factory_.reset();
    dev_set_.reset();
    dev_pi_ = compute::vector<Float>();
    dev_beta_ = compute::vector<Float>();
    ContextTest::TearDown();
  }
  
  uint32_t N_;
  uint32_t K_;
  uint32_t num_tries_;
  compute::vector<Edge> dev_edges_;
  std::shared_ptr<OpenClSetFactory> factory_;
  std::unique_ptr<OpenClSet> dev_set_;
  compute::vector<Float> dev_pi_;
  compute::vector<Float> dev_beta_;
  Config cfg_;
};

TEST_P(WgPerplexityTest, Equal) {
  cfg_.ppx_wg_size = GetParam();
  mcmc::PerplexityCalculator ppxSimple(mcmc::PerplexityCalculator::EDGE_PER_THREAD,
                                 cfg_, queue_, dev_beta_, dev_pi_, dev_edges_,
                                 dev_set_.get(), MakeCompileFlags(cfg_),
                                 Learner::GetBaseFuncs());
  Float error = 0.15;
  Float ppx1 = ppxSimple();
  double ppx1_time = 0;
  double ppx1_total_time = 0;
  for (uint32_t i = 0; i < num_tries_; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    ASSERT_NEAR(ppx1, ppxSimple(), error);
    auto t2 = std::chrono::high_resolution_clock::now();
    ppx1_time += ppxSimple.LastInvocationTime();
    ppx1_total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  }
  mcmc::PerplexityCalculator ppxWg(mcmc::PerplexityCalculator::EDGE_PER_WORKGROUP,
                                 cfg_, queue_, dev_beta_, dev_pi_, dev_edges_,
                                 dev_set_.get(), MakeCompileFlags(cfg_),
                                 Learner::GetBaseFuncs());
  Float ppx2 = ppxWg();
  double ppx2_time = 0;
  double ppx2_total_time = 0;
  ASSERT_NEAR(ppx1, ppx2, error);
  for (uint32_t i = 0; i < num_tries_; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    ASSERT_NEAR(ppx2, ppxWg(), error);
    auto t2 = std::chrono::high_resolution_clock::now();
    ppx2_time += ppxWg.LastInvocationTime();
    ppx2_total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
  }
  LOG(INFO) << "K=" << K_ << ", WG=" << cfg_.ppx_wg_size
    << ", EDGE_PER_THREAD=" << ppx1_time / num_tries_ << " (" << 100*ppx1_time / ppx1_total_time << "%)"
    << ", EDGE_PER_WG=" << ppx2_time / num_tries_ << " (" << 100*ppx2_time / ppx2_total_time << "%)"
    << " (" << (ppx1_time / ppx2_time) << "x)";
}
INSTANTIATE_TEST_CASE_P(WorkGroups, WgPerplexityTest,
                        ::testing::ValuesIn(std::vector<uint32_t>(
                            {32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
