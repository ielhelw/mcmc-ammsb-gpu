#include "mcmc/learner.h"

#include <algorithm>
#include <glog/logging.h>
#include <random>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string& Learner::GetBaseFuncs() {
  static const std::string kSourceBaseFuncs =
      GetSourceGuard() + GetClTypes() + "\n" + OpenClSetFactory::GetHeader() +
      "\n" + BOOST_COMPUTE_STRINGIZE_SOURCE(

                 typedef VERTEX_TYPE Vertex; typedef EDGE_TYPE Edge;

                 inline Vertex Vertex0(Edge e) {
                   return (Vertex)((e & 0xffff0000) >> 32);
                 }

                     inline Vertex Vertex1(Edge e) {
                       return (Vertex)((e & 0x0000ffff));
                     }

                     inline Edge MakeEdge(
                         Vertex u, Vertex v) { return (((Edge)u) << 32) | v; }

                     inline __global Float *
                     Pi(__global Float * pi, Vertex u) { return pi + u * K; }

                     inline Float get_eps_t(uint step_count) {
                   return EPS_A * pow(1 + step_count / EPS_B, -EPS_C);
                 }

                 );
  return kSourceBaseFuncs;
}

Learner::Learner(const Config& cfg, compute::command_queue queue)
    : cfg_(cfg),
      queue_(queue),
      beta_(cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      allocFactory_(RowPartitionedMatrixFactory<Float>::New(queue_)),
      pi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
      phi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
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
                         beta_, pi_.get(), heldoutEdges_, heldoutSet_.get(),
                         compileFlags_, GetBaseFuncs()),
      phiUpdater_(PhiUpdater::NODE_PER_WORKGROUP, cfg_, queue_, beta_, pi_.get(),
                  phi_.get(), trainingSet_.get(), compileFlags_, GetBaseFuncs()) {
  // gamma generator
  std::mt19937 mt19937;
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  GenerateAndNormalize(&queue_, &gamma, &theta_, &beta_, 1, cfg_.K);
  GenerateAndNormalize(&queue_, &gamma, phi_.get(), pi_.get());
}

void Learner::run() {
  uint32_t step_count = 1;
  for (; step_count < 3; ++step_count) {
    LOG(INFO) << "ppx[" << step_count << "] = " << heldoutPerplexity_();
    std::set<uint> set;
    while (set.size() < cfg_.mini_batch_size) {
      set.insert(rand() % cfg_.N);
    }
    std::vector<uint> hmbn(set.begin(), set.end());
    std::vector<uint> hn(hmbn.size() * cfg_.num_node_sample);
    std::generate(hn.begin(), hn.end(), [this]() { return rand() % cfg_.N; });
    compute::vector<uint> mbn(hmbn.begin(), hmbn.end(), queue_);
    compute::vector<uint> n(hn.begin(), hn.end(), queue_);
    phiUpdater_(mbn, n, static_cast<uint32_t>(mbn.size()));
  }
}

}  // namespace mcmc
