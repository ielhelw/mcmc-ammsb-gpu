#include <gtest/gtest.h>
#include <glog/logging.h>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>
#include <chrono>

#include "mcmc/test.h"
#include "mcmc/phi.h"
#include "mcmc/learner.h"

namespace mcmc {
namespace test {

class WgPhiTest : public ContextTest,
                         public ::testing::WithParamInterface<uint32_t> {
 protected:
  WgPhiTest(uint32_t K = 1024, uint32_t N = 10*1024)
    : num_tries_(3) {
    cfg_.N = N;
    cfg_.K = K;
    cfg_.phi_wg_size = 32;
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
    // init phi_pi
    std::mt19937 mt19937;
    std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
    auto gamma = std::bind(gamma_distribution, mt19937);
    Learner::GenerateAndNormalize(&queue_, &gamma, &theta_, &beta_, 1, cfg_.K);
    Learner::GenerateAndNormalize(&queue_, &gamma, phi_.get(), pi_.get());
  }

  void SetUp() override {
    ContextTest::SetUp();
    std::vector<Edge> edges = GenerateRandomEdges(32*cfg_.N);
    Set set(edges.size());
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      ASSERT_TRUE(set.Insert(*it));
    }
    factory_ = OpenClSetFactory::New(queue_);
    dev_set_.reset(factory_->CreateSet(set.Serialize()));
    theta_ = compute::vector<Float>(2 * cfg_.K, queue_.get_context());
    beta_ = compute::vector<Float>(cfg_.K, queue_.get_context());
    allocFactory = RowPartitionedMatrixFactory<Float>::New(queue_);
    phi_.reset(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    ASSERT_EQ(1, phi_->Blocks().size());
    pi_.reset(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    ASSERT_EQ(1, pi_->Blocks().size());
  }
  
  void TearDown() override {
    factory_.reset();
    dev_set_.reset();
    allocFactory.reset();
    pi_.reset();
    phi_.reset();
    theta_ = compute::vector<Float>();
    beta_ = compute::vector<Float>();
    ContextTest::TearDown();
  }

  void Run(PhiUpdater::Mode mode) {
    Float delta = 1.0 / cfg_.K;
    PhiUpdater updater(mode, cfg_, queue_,
        beta_, pi_.get(), phi_.get(), dev_set_.get(),
        MakeCompileFlags(cfg_), Learner::GetBaseFuncs());
    // generate random mini-batch nodes
    std::vector<uint> hmbn(cfg_.N);
    uint32_t node = 0;
    std::generate(hmbn.begin(), hmbn.end(), [&node]() { return node++; });
    std::vector<uint> hn(hmbn.size() * cfg_.num_node_sample);
    srand(42);
    std::generate(hn.begin(), hn.end(), [this]() { return rand() % cfg_.N; });
    compute::vector<uint> mbn(hmbn.begin(), hmbn.end(), queue_);
    compute::vector<uint> n(hn.begin(), hn.end(), queue_);
    std::vector<Float> host_pi_tmp(cfg_.N * cfg_.K);
    std::vector<Float> host_pi(cfg_.N * cfg_.K);
    double time = 0;
    for (uint32_t i = 0; i < num_tries_; ++i) {
      Reset();
      updater(mbn, n, static_cast<uint32_t>(mbn.size()));
      time += updater.LastInvocationTime();
      if (i == 0) {
        compute::copy(pi_->Blocks()[0].begin(), pi_->Blocks()[0].end(), host_pi.begin(), queue_);
      } else {
        compute::copy(pi_->Blocks()[0].begin(), pi_->Blocks()[0].end(), host_pi_tmp.begin(), queue_);
      }
    }
    LOG(INFO) << "WG=" << cfg_.phi_wg_size << ", nano=" << time / num_tries_;
  }
  
  uint32_t num_tries_;
  std::shared_ptr<OpenClSetFactory> factory_;
  std::unique_ptr<OpenClSet> dev_set_;
  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;
  std::unique_ptr<RowPartitionedMatrix<Float>> phi_;
  compute::vector<Float> beta_;
  compute::vector<Float> theta_;
  Config cfg_;
};

TEST_P(WgPhiTest, VerifyModes) {
  num_tries_ = 1;
  cfg_.phi_wg_size = GetParam();
  cfg_.phi_disable_noise = true;
  Run(PhiUpdater::NODE_PER_THREAD);
  std::vector<Float> host_phi1(cfg_.N * cfg_.K);
  compute::copy(phi_->Blocks()[0].begin(), phi_->Blocks()[0].end(), host_phi1.begin(), queue_);
  std::vector<Float> host_pi1(cfg_.N * cfg_.K);
  compute::copy(pi_->Blocks()[0].begin(), pi_->Blocks()[0].end(), host_pi1.begin(), queue_);

  Run(PhiUpdater::NODE_PER_WORKGROUP);
  std::vector<Float> host_phi2(cfg_.N * cfg_.K);
  compute::copy(phi_->Blocks()[0].begin(), phi_->Blocks()[0].end(), host_phi2.begin(), queue_);
  std::vector<Float> host_pi2(cfg_.N * cfg_.K);
  compute::copy(pi_->Blocks()[0].begin(), pi_->Blocks()[0].end(), host_pi2.begin(), queue_);

  for (uint32_t k = 0; k < host_phi1.size(); ++k) {
    ASSERT_NEAR(host_phi1[k], host_phi2[k], std::max(1e-5, 0.02 * std::abs(host_phi1[k])));
  }
  for (uint32_t k = 0; k < host_phi1.size(); ++k) {
    ASSERT_NEAR(host_pi1[k], host_pi2[k], std::max(1e-5, 0.02 * std::abs(host_pi1[k])));
  }
}

TEST_P(WgPhiTest, NodePerThread) {
  cfg_.phi_wg_size = GetParam();
  Run(PhiUpdater::NODE_PER_THREAD);
}

TEST_P(WgPhiTest, NodePerWorkGroup) {
  cfg_.phi_wg_size = GetParam();
  Run(PhiUpdater::NODE_PER_WORKGROUP);
}

INSTANTIATE_TEST_CASE_P(WorkGroups, WgPhiTest,
                        ::testing::ValuesIn(std::vector<uint32_t>(
                            {32, 64, 128, 256})));
//                            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
