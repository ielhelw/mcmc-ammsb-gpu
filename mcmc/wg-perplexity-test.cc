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
    dev_edges_.reset(new clcuda::Buffer<Edge>(*context_, *queue_, edges.begin(),
                                              edges.end()));
    Set set(edges.size());
    ASSERT_TRUE(set.SetContents(edges.begin(), edges.end()));
    factory_ = OpenClSetFactory::New(*queue_);
    dev_set_.reset(factory_->CreateSet(set));
    std::mt19937 mt19937;
    std::gamma_distribution<Float> gamma_dist(1, 1);
    auto gen = std::bind(gamma_dist, mt19937);
    std::vector<Float> pi(N_ * K_);
    std::generate(pi.begin(), pi.end(), gen);
    std::vector<Float> beta(2 * K_);
    std::generate(beta.begin(), beta.end(), gen);
    allocFactory = (RowPartitionedMatrixFactory<Float>::New(*queue_));
    dev_pi_.reset(allocFactory->CreateMatrix(N_, K_));
    ASSERT_EQ(static_cast<size_t>(1), dev_pi_->Blocks().size());
    ASSERT_EQ(pi.size(), dev_pi_->Blocks()[0].GetSize() / sizeof(Float));
    dev_pi_->Blocks()[0].Write(*queue_, pi.size(), pi);
    clcuda::Buffer<Float> phi(*context_, N_);
    mcmc::algorithm::PartitionedNormalizer<Float>(*queue_, dev_pi_.get(), phi,
                                                  K_)();
    dev_beta_.reset(new clcuda::Buffer<Float>(*context_, *queue_, beta.begin(),
                                              beta.end()));
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
  std::vector<double> ppxs1;
  for (uint32_t i = 0; i < num_tries_; ++i) {
    ppxs1.push_back(ppxSimple());
  }
  mcmc::PerplexityCalculator ppxWg(
      mcmc::PerplexityCalculator::EDGE_PER_WORKGROUP, cfg_, *queue_, *dev_beta_,
      dev_pi_.get(), *dev_edges_, dev_set_.get(), MakeCompileFlags(cfg_),
      Learner::GetBaseFuncs());
  std::vector<double> ppxs2;
  for (uint32_t i = 0; i < num_tries_; ++i) {
    ppxs2.push_back(ppxWg());
    ASSERT_NEAR(ppxs1[i], ppxs2[i], /*INCREASE ERROREVERY ITERATION */(i + 1) *
                                        error * std::abs(ppxs1[i]));
  }
}
INSTANTIATE_TEST_CASE_P(
    WorkGroups, WgPerplexityTest,
    ::testing::ValuesIn(std::vector<uint32_t>({32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
