#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>

#include "mcmc/test.h"
#include "mcmc/beta.h"
#include "mcmc/learner.h"

namespace mcmc {
namespace test {

class WgBetaTest : public ContextTest,
                   public ::testing::WithParamInterface<uint32_t> {
 protected:
  WgBetaTest(uint32_t K = 1024, uint32_t N = 4 * 1024) : num_tries_(3) {
    cfg_.N = N;
    cfg_.K = K;
    cfg_.mini_batch_size = N;
    cfg_.num_node_sample = 64;
    cfg_.eta0 = 1;
    cfg_.eta1 = 1;
  }

  std::vector<Edge> GenerateRandomEdges(uint32_t num_edges) {
    std::vector<Edge> edges(num_edges);
    std::default_random_engine generator;
    std::uniform_int_distribution<Vertex> distribution(0, cfg_.N);
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

  void Reset() {
    std::mt19937 mt19937(42);
    std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
    auto gamma = std::bind(gamma_distribution, mt19937);
    random::RandomAndNormalize(queue_.get(), &gamma, theta_.get(), beta_.get(),
                               2);
  }

  void SetUp() override {
    ContextTest::SetUp();
    std::vector<Edge> edges = GenerateRandomEdges(32 * cfg_.N);
    Set set(edges.size());
    ASSERT_TRUE(set.SetContents(edges.begin(), edges.end()));
    factory_ = OpenClSetFactory::New(*queue_);
    dev_set_.reset(factory_->CreateSet(set));
    theta_.reset(new clcuda::Buffer<Float>(*context_, 2 * cfg_.K));
    beta_.reset(new clcuda::Buffer<Float>(*context_, 2 * cfg_.K));
    allocFactory = RowPartitionedMatrixFactory<Float>::New(*queue_);
    pi_.reset(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    clcuda::Buffer<Float> phi(*context_, cfg_.N);
    random::RandomGammaAndNormalize(queue_.get(), cfg_.eta0, cfg_.eta1,
                                    pi_.get(), &phi);
  }

  void TearDown() override {
    updater_.reset();
    factory_.reset();
    dev_set_.reset();
    allocFactory.reset();
    pi_.reset();
    theta_.reset();
    beta_.reset();
    ContextTest::TearDown();
  }

  void Run(BetaUpdater::Mode mode) {
    std::vector<Float> host_theta(theta_->GetSize() / sizeof(Float));
    std::vector<Float> host_theta_sum(cfg_.K);
    updater_.reset(new BetaUpdater(
        mode, cfg_, *queue_, *theta_, *beta_, pi_.get(), dev_set_.get(),
        MakeCompileFlags(cfg_), Learner::GetBaseFuncs()));
    std::vector<Edge> random_edges = GenerateRandomEdges(1024);
    clcuda::Buffer<Edge> edges(*context_, *queue_, random_edges.begin(),
                               random_edges.end());
    for (uint32_t i = 0; i < num_tries_; ++i) {
      Reset();
      theta_->Read(*queue_, host_theta.size(), host_theta);
      (*updater_)(&edges, edges.GetSize() / sizeof(Edge), 0.01);
      updater_->GetThetaSum().Read(*queue_, host_theta_sum.size(),
                                   host_theta_sum);
    }
  }

  Config cfg_;
  std::unique_ptr<clcuda::Buffer<Float>> theta_;
  std::unique_ptr<clcuda::Buffer<Float>> beta_;
  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;
  std::shared_ptr<OpenClSetFactory> factory_;
  std::unique_ptr<OpenClSet> dev_set_;
  uint32_t num_tries_;
  std::unique_ptr<BetaUpdater> updater_;
};

TEST_P(WgBetaTest, VerifyModes) {
  cfg_.beta_wg_size = GetParam();
  num_tries_ = 1;

  LOG(INFO) << "CALL 1";
  Run(BetaUpdater::EDGE_PER_THREAD);
  std::vector<Float> theta_sum1(updater_->GetThetaSum().GetSize() /
                                sizeof(Float));
  updater_->GetThetaSum().Read(*queue_, theta_sum1.size(), theta_sum1);
  std::vector<Float> grads1(2 * cfg_.K);
  updater_->GetGrads().Read(*queue_, 2 * cfg_.K, grads1);
  std::vector<Float> theta1(theta_->GetSize() / sizeof(Float));
  theta_->Read(*queue_, theta1.size(), theta1);

  LOG(INFO) << "CALL 2";
  Run(BetaUpdater::EDGE_PER_WORKGROUP);
  std::vector<Float> theta_sum2(updater_->GetThetaSum().GetSize() /
                                sizeof(Float));
  updater_->GetThetaSum().Read(*queue_, theta_sum2.size(), theta_sum2);
  std::vector<Float> grads2(2 * cfg_.K);
  updater_->GetGrads().Read(*queue_, 2 * cfg_.K, grads2);
  std::vector<Float> theta2(theta_->GetSize() / sizeof(Float));
  theta_->Read(*queue_, theta2.size(), theta2);

  ASSERT_EQ(theta_sum1.size(), theta_sum2.size());
  for (uint32_t k = 0; k < theta_sum1.size(); ++k) {
    ASSERT_NEAR(theta_sum1[k], theta_sum2[k],
                std::max(0.00001, 0.02 * std::abs(theta_sum1[k])));
  }

  ASSERT_EQ(theta1.size(), theta2.size());
  for (uint32_t k = 0; k < theta1.size(); ++k) {
    ASSERT_NEAR(theta1[k], theta2[k],
                std::max(0.00001, 0.02 * std::abs(theta1[k])));
  }
}

TEST_P(WgBetaTest, EdgePerThread) {
  cfg_.beta_wg_size = GetParam();
  Run(BetaUpdater::EDGE_PER_THREAD);
}

TEST_P(WgBetaTest, EdgePerWorkGroup) {
  cfg_.beta_wg_size = GetParam();
  Run(BetaUpdater::EDGE_PER_WORKGROUP);
}

INSTANTIATE_TEST_CASE_P(
    WorkGroups, WgBetaTest,
    ::testing::ValuesIn(std::vector<uint32_t>({32, 64, 128, 256, 512, 1024})));
//                            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
