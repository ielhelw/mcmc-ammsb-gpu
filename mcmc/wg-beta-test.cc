#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>
#include <chrono>

#include "mcmc/test.h"
#include "mcmc/beta.h"
#include "mcmc/learner.h"

namespace mcmc {
namespace test {

class WgBetaTest : public ContextTest,
                   public ::testing::WithParamInterface<uint32_t> {
 protected:
  WgBetaTest(uint32_t K = 1024, uint32_t N = 10 * 1024) : num_tries_(3) {
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
    Learner::GenerateAndNormalize(&queue_, &gamma, &theta_, &beta_, 1, cfg_.K);
  }

  void SetUp() override {
    ContextTest::SetUp();
    std::vector<Edge> edges = GenerateRandomEdges(32 * cfg_.N);
    Set set(edges.size());
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      ASSERT_TRUE(set.Insert(*it));
    }
    factory_ = OpenClSetFactory::New(queue_);
    dev_set_.reset(factory_->CreateSet(set.Serialize()));
    theta_ = compute::vector<Float>(2 * cfg_.K, queue_.get_context());
    beta_ = compute::vector<Float>(cfg_.K, queue_.get_context());
    allocFactory = RowPartitionedMatrixFactory<Float>::New(queue_);
    pi_.reset(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    std::unique_ptr<RowPartitionedMatrix<Float>> phi(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    std::mt19937 mt19937(24);
    std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
    auto gamma = std::bind(gamma_distribution, mt19937);
    Learner::GenerateAndNormalize(&queue_, &gamma, phi.get(), pi_.get());
  }

  void TearDown() override {
    updater_.reset();
    factory_.reset();
    dev_set_.reset();
    allocFactory.reset();
    pi_.reset();
    theta_ = compute::vector<Float>();
    beta_ = compute::vector<Float>();
    ContextTest::TearDown();
  }

  void Run(BetaUpdater::Mode mode) {
    std::vector<Float> host_theta(theta_.size());
    std::vector<Float> host_theta_sum(cfg_.K);
    updater_.reset(new BetaUpdater(mode, cfg_, queue_, theta_, beta_, pi_.get(), dev_set_.get(),
                        MakeCompileFlags(cfg_), Learner::GetBaseFuncs()));
    std::vector<Edge> random_edges = GenerateRandomEdges(1024);
    compute::vector<Edge> edges(random_edges.begin(), random_edges.end(),
                                queue_);
    double time = 0;
    for (uint32_t i = 0; i < num_tries_; ++i) {
      Reset();
      compute::copy(theta_.begin(), theta_.end(), host_theta.begin(), queue_);
      (*updater_)(&edges, edges.size(), 0.01);
      time += updater_->LastInvocationTime();
      compute::copy(updater_->GetThetaSum().begin(), updater_->GetThetaSum().end(),
                    host_theta_sum.begin(), queue_);
      for (uint32_t k = 0; k < cfg_.K; ++k) {
        ASSERT_FLOAT_EQ(host_theta[k] + host_theta[cfg_.K + k],
                        host_theta_sum[k]);
      }
    }
    LOG(INFO) << "WG=" << cfg_.beta_wg_size << ", nano=" << time / num_tries_;
  }

  Config cfg_;
  compute::vector<Float> theta_;
  compute::vector<Float> beta_;
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
  
  Run(BetaUpdater::EDGE_PER_THREAD);
  std::vector<Float> theta_sum1(updater_->GetThetaSum().size());
  compute::copy(updater_->GetThetaSum().begin(), updater_->GetThetaSum().end(),
      theta_sum1.begin(), queue_);
  std::vector<Float> grads1(2 * cfg_.K);
  compute::copy(updater_->GetGrads().begin(), updater_->GetGrads().begin() + 2*cfg_.K, grads1.begin(), queue_);
  std::vector<Float> theta1(theta_.size());
  compute::copy(theta_.begin(), theta_.end(), theta1.begin(), queue_);

  Run(BetaUpdater::EDGE_PER_WORKGROUP);
  std::vector<Float> theta_sum2(updater_->GetThetaSum().size());
  compute::copy(updater_->GetThetaSum().begin(), updater_->GetThetaSum().end(),
      theta_sum2.begin(), queue_);
  std::vector<Float> grads2(2 * cfg_.K);
  compute::copy(updater_->GetGrads().begin(), updater_->GetGrads().begin() + 2*cfg_.K, grads2.begin(), queue_);
  std::vector<Float> theta2(theta_.size());
  compute::copy(theta_.begin(), theta_.end(), theta2.begin(), queue_);
  
  ASSERT_EQ(theta_sum1.size(), theta_sum2.size());
  for (uint32_t k = 0; k < theta_sum1.size(); ++k) {
    ASSERT_NEAR(theta_sum1[k], theta_sum2[k], 0.000001);
  }
  
  for (uint32_t k = 0; k < grads1.size(); ++k) {
    ASSERT_NEAR(grads1[k], grads2[k], std::max(0.00001, 0.001*std::abs(grads1[k])));
  }
 
  ASSERT_EQ(theta1.size(), theta2.size());
  for (uint32_t k = 0; k < theta1.size(); ++k) {
    ASSERT_NEAR(theta1[k], theta2[k], 0.00001);
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
