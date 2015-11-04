#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>

#include "mcmc/test.h"
#include "mcmc/phi.h"
#include "mcmc/learner.h"

namespace mcmc {
namespace test {

class WgPhiTest : public ContextTest,
                  public ::testing::WithParamInterface<uint32_t> {
 protected:
  WgPhiTest(uint32_t K = 512, uint32_t N = 4 * 1024) : num_tries_(3) {
    cfg_.N = N;
    cfg_.K = K;
    cfg_.phi_wg_size = 32;
    cfg_.mini_batch_size = N;
    cfg_.num_node_sample = 4;
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
    random::RandomAndNormalize(queue_.get(), &gamma, theta_.get(), beta_.get(),
                               2);
    random::RandomGammaAndNormalize(queue_.get(), cfg_.eta0, cfg_.eta1,
                                    pi_.get(), phi_.get());
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
    phi_.reset(new clcuda::Buffer<Float>(*context_, cfg_.N));
    pi_.reset(allocFactory->CreateMatrix(cfg_.N, cfg_.K));
    ASSERT_EQ(static_cast<size_t>(1), pi_->Blocks().size());
  }

  void TearDown() override {
    factory_.reset();
    dev_set_.reset();
    allocFactory.reset();
    pi_.reset();
    phi_.reset();
    theta_.reset();
    beta_.reset();
    ContextTest::TearDown();
  }

  void Run(PhiUpdater::Mode mode) {
    PhiUpdater updater(mode, cfg_, *queue_, *beta_, pi_.get(), *phi_,
                       dev_set_.get(), MakeCompileFlags(cfg_),
                       Learner::GetBaseFuncs());
    // generate random mini-batch nodes
    std::vector<uint> hmbn(cfg_.N);
    uint32_t node = 0;
    std::generate(hmbn.begin(), hmbn.end(), [&node]() { return node++; });
    std::vector<uint> hn(hmbn.size() * cfg_.num_node_sample);
    srand(42);
    std::generate(hn.begin(), hn.end(), [this]() { return rand() % cfg_.N; });
    clcuda::Buffer<uint> mbn(*context_, *queue_, hmbn.begin(), hmbn.end());
    clcuda::Buffer<uint> n(*context_, *queue_, hn.begin(), hn.end());
    std::vector<Float> host_pi_tmp(cfg_.N * cfg_.K);
    std::vector<Float> host_pi(cfg_.N * cfg_.K);
    for (uint32_t i = 0; i < num_tries_; ++i) {
      Reset();
      updater(mbn, n, static_cast<uint32_t>(hmbn.size()));
      if (i == 0) {
        pi_->Blocks()[0].Read(*queue_, host_pi.size(), host_pi);
      } else {
        pi_->Blocks()[0].Read(*queue_, host_pi_tmp.size(), host_pi_tmp);
      }
    }
  }

  uint32_t num_tries_;
  std::shared_ptr<OpenClSetFactory> factory_;
  std::unique_ptr<OpenClSet> dev_set_;
  std::shared_ptr<RowPartitionedMatrixFactory<Float>> allocFactory;
  std::unique_ptr<RowPartitionedMatrix<Float>> pi_;
  std::unique_ptr<clcuda::Buffer<Float>> phi_;
  std::unique_ptr<clcuda::Buffer<Float>> beta_;
  std::unique_ptr<clcuda::Buffer<Float>> theta_;
  Config cfg_;
};

TEST_P(WgPhiTest, VerifyModes) {
  num_tries_ = 1;
  cfg_.phi_wg_size = GetParam();
  cfg_.phi_disable_noise = true;
  Run(PhiUpdater::NODE_PER_THREAD);
  std::vector<Float> host_phi1(cfg_.N);
  phi_->Read(*queue_, host_phi1.size(), host_phi1);
  std::vector<Float> host_pi1(cfg_.N * cfg_.K);
  pi_->Blocks()[0].Read(*queue_, host_pi1.size(), host_pi1);

  Run(PhiUpdater::NODE_PER_WORKGROUP);
  std::vector<Float> host_phi2(cfg_.N);
  phi_->Read(*queue_, host_phi2.size(), host_phi2);
  std::vector<Float> host_pi2(cfg_.N * cfg_.K);
  pi_->Blocks()[0].Read(*queue_, host_pi2.size(), host_pi2);

  for (uint32_t k = 0; k < host_phi1.size(); ++k) {
    ASSERT_NEAR(host_phi1[k], host_phi2[k],
                0.02 * std::abs(host_phi1[k]));
  }
  for (uint32_t k = 0; k < host_phi1.size(); ++k) {
    ASSERT_NEAR(host_pi1[k], host_pi2[k],
                0.02 * std::abs(host_pi1[k]));
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
                        ::testing::ValuesIn(std::vector<uint32_t>({32,  64,
                                                                   128, 256})));
//                            {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024})));

}  // namespace test
}  // namespace mcmc
