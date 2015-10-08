#include "mcmc/learner.h"

#include <algorithm>
#include <glog/logging.h>
#include <random>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string Learner::kSourceBaseFuncs =
    GetSourceGuard() + "\n" +
    OpenClSetFactory::GetHeader() + "\n" +
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        typedef FLOAT_TYPE Float; typedef VERTEX_TYPE Vertex;
        typedef EDGE_TYPE Edge;

        inline Vertex Vertex0(Edge e) {
          return (Vertex)((e & 0xffff0000) >> 32);
        }

            inline Vertex Vertex1(Edge e) { return (Vertex)((e & 0x0000ffff)); }

            inline __global Float *
            Pi(__global Float * pi, Vertex u) { return pi + u * K; }

            inline Float get_eps_t(uint step_count) {
          return EPS_A * pow(1 + step_count / EPS_B, -EPS_C);
        }

        );

Learner::Learner(const Config& cfg, compute::command_queue queue)
    : cfg_(cfg),
      queue_(queue),
      beta_(cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      pi_(cfg_.N * cfg_.K, queue_.get_context()),
      phi_(cfg_.N * cfg_.K, queue_.get_context()),
      scratch_(queue_.get_context(), cfg_.N * cfg_.K * sizeof(Float)),
      setFactory_(OpenClSetFactory::New(queue_)),
      trainingSet_(setFactory_->CreateSet(cfg_.training->Serialize())),
      heldoutSet_(setFactory_->CreateSet(cfg_.heldout->Serialize())),
      trainingEdges_(cfg_.training_edges.begin(), cfg_.training_edges.end(),
                     queue_),
      heldoutEdges_(cfg_.heldout_edges.begin(), cfg_.heldout_edges.end(),
                    queue_),
      compileFlags_(MakeCompileFlags(cfg_)),
      heldoutPerplexity_(PerplexityCalculator::EDGE_PER_WORKGROUP, cfg_, queue_,
                         beta_, pi_, heldoutEdges_, heldoutSet_.get(),
                         compileFlags_, kSourceBaseFuncs) {
  // gamma generator
  std::mt19937 mt19937;
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  // generate theta
  std::vector<Float> host_theta(theta_.size());
  std::generate(host_theta.begin(), host_theta.end(), gamma);
  compute::copy(host_theta.begin(), host_theta.end(), theta_.begin(), queue_);
  compute::copy(theta_.begin(), theta_.begin() + cfg_.K, beta_.begin(), queue_);
  mcmc::algorithm::Normalizer<Float>(queue_, &beta_, cfg_.K, 32)();
  // generate phi
  std::vector<Float> host_phi(phi_.size());
  std::generate(host_phi.begin(), host_phi.end(), gamma);
  compute::copy(host_phi.begin(), host_phi.end(), phi_.begin(), queue_);
  compute::copy(phi_.begin(), phi_.end(), pi_.begin(), queue_);
  mcmc::algorithm::Normalizer<Float>(queue_, &pi_, cfg_.K, 32)();
}

void Learner::run() {
  uint32_t step_count = 1;
  for (; step_count < 10; ++step_count) {
    LOG(INFO) << "ppx[" << step_count << "] = " << heldoutPerplexity_();
  }
}

}  // namespace mcmc
