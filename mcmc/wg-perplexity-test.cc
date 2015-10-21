#include <gtest/gtest.h>
#include <glog/logging.h>

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
      : N_(N), K_(K), num_tries_(3) {}

  void SetUp() override {
    ContextTest::SetUp();
    std::vector<Edge> edges = GenerateRandomEdges(1024, N_ - 1);
    dev_edges_.reset(new clcuda::Buffer<Edge>(*context_, *queue_, edges.begin(), edges.end()));
    Set set(edges.size());
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      ASSERT_TRUE(set.Insert(*it));
    }
    factory_ = OpenClSetFactory::New(*queue_);
    dev_set_.reset(factory_->CreateSet(set.Serialize()));
    std::mt19937 mt19937;
    std::gamma_distribution<Float> gamma_dist(1, 1);
    auto gen = std::bind(gamma_dist, mt19937);
    std::vector<Float> pi(N_ * K_);
    std::generate(pi.begin(), pi.end(), gen);
    std::vector<Float> beta(2 * K_);
    std::generate(beta.begin(), beta.end(), gen);
    allocFactory = (RowPartitionedMatrixFactory<Float>::New(*queue_));
    dev_pi_.reset(allocFactory->CreateMatrix(N_, K_));
    ASSERT_EQ(1, dev_pi_->Blocks().size());
    ASSERT_EQ(pi.size(), dev_pi_->Blocks()[0].GetSize() / sizeof(Float));
    dev_pi_->Blocks()[0].Write(*queue_, pi.size(), pi);
    clcuda::Buffer<Float> phi(*context_, N_);
    mcmc::algorithm::PartitionedNormalizer<Float>(*queue_, dev_pi_.get(), phi,
                                                  K_)();
    dev_beta_.reset(new clcuda::Buffer<Float>(*context_, *queue_, beta.begin(), beta.end()));
    mcmc::algorithm::Normalizer<Float>(*queue_, dev_beta_.get(), 2, 1)();
    cfg_.K = K_;
  }

  void TearDown() override {
    dev_edges_.reset();
    factory_.reset();
    dev_set_.reset();
    dev_pi_.reset();
    allocFactory.reset();
    dev_beta_.reset();
    ContextTest::TearDown();
  }

  uint32_t N_;
  uint32_t K_;
  uint32_t num_tries_;
  std::unique_ptr<clcuda::Buffer<Edge>> dev_edges_;
  std::shared_ptr<OpenClSetFactory> factory_;
  std::unique_ptr<OpenClSet> dev_set_;
  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory;
  std::unique_ptr<RowPartitionedMatrix<Float>> dev_pi_;
  std::unique_ptr<clcuda::Buffer<Float>> dev_beta_;
  Config cfg_;
};

TEST_P(WgPerplexityTest, Equal) {
  num_tries_ = 1;
  cfg_.ppx_wg_size = GetParam();
  mcmc::PerplexityCalculator ppxSimple(
      mcmc::PerplexityCalculator::EDGE_PER_THREAD, cfg_, *queue_, *dev_beta_,
      dev_pi_.get(), *dev_edges_, dev_set_.get(), MakeCompileFlags(cfg_),
      Learner::GetBaseFuncs());
  Float error = 0.05;
  double ppx1_time = 0;
  double ppx1_total_time = 0;
  std::vector<double> ppxs1;
  for (uint32_t i = 0; i < num_tries_; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    ppxs1.push_back(ppxSimple());
    auto t2 = std::chrono::high_resolution_clock::now();
    ppx1_time += ppxSimple.LastInvocationTime();
    ppx1_total_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
  mcmc::PerplexityCalculator ppxWg(
      mcmc::PerplexityCalculator::EDGE_PER_WORKGROUP, cfg_, *queue_, *dev_beta_,
      dev_pi_.get(), *dev_edges_, dev_set_.get(), MakeCompileFlags(cfg_),
      Learner::GetBaseFuncs());
  double ppx2_time = 0;
  double ppx2_total_time = 0;
  std::vector<double> ppxs2;
  for (uint32_t i = 0; i < num_tries_; ++i) {
    auto t1 = std::chrono::high_resolution_clock::now();
    ppxs2.push_back(ppxWg());
    auto t2 = std::chrono::high_resolution_clock::now();
    ASSERT_NEAR(ppxs1[i], ppxs2[i], /*INCREASE ERROREVERY ITERATION */ (i + 1) *
                                        error * std::abs(ppxs1[i]));
    ppx2_time += ppxWg.LastInvocationTime();
    ppx2_total_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
  LOG(INFO) << "K=" << K_ << ", WG=" << cfg_.ppx_wg_size
            << ", EDGE_PER_THREAD=" << ppx1_time / num_tries_ << " ("
            << 100 * ppx1_time / ppx1_total_time << "%)"
            << ", EDGE_PER_WG=" << ppx2_time / num_tries_ << " ("
            << 100 * ppx2_time / ppx2_total_time << "%)"
            << " (" << (ppx1_time / ppx2_time) << "x)";
}
INSTANTIATE_TEST_CASE_P(
    WorkGroups, WgPerplexityTest,
    ::testing::ValuesIn(std::vector<uint32_t>({32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
