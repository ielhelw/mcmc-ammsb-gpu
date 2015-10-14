#include "mcmc/learner.h"

#include <algorithm>
#include <boost/compute/algorithm/reduce.hpp>
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
                   return (Vertex)((e & 0xffffffff00000000) >> 32);
                 }

                     inline Vertex Vertex1(Edge e) {
                       return (Vertex)((e & 0x00000000ffffffff));
                     }

                     inline Edge MakeEdge(Vertex u, Vertex v) {
                       return (((Edge)u) << 32) | v;
                     }

                     inline Float Beta(__global Float* g_beta, uint k) { return g_beta[k<<1]; }
                     
                     inline Float Theta0(__global Float* g_theta, uint k) { return g_theta[k<<1]; }
                     
                     inline Float Theta1(__global Float* g_theta, uint k) { return g_theta[k<<1 + 1]; }
                     
                     inline void SetTheta0(__global Float* g_theta, uint k, Float v) { g_theta[k<<1] = v; }
                     
                     inline void SetTheta1(__global Float* g_theta, uint k, Float v) { g_theta[k<<1 + 1] = v; }

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
      beta_(2 * cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      allocFactory_(RowPartitionedMatrixFactory<Float>::New(queue_)),
      pi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
      phi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
      setFactory_(OpenClSetFactory::New(queue_)),
      trainingSet_(setFactory_->CreateSet(cfg_.training->Serialize())),
      heldoutSet_(setFactory_->CreateSet(cfg_.heldout->Serialize())),
      trainingEdges_(cfg_.training_edges.begin(), cfg_.training_edges.end(),
                     queue_),
      heldoutEdges_(cfg_.heldout_edges.begin(), cfg_.heldout_edges.end(),
                    queue_),
      compileFlags_(MakeCompileFlags(cfg_)),
      heldoutPerplexity_((queue_.get_device().type() == CL_DEVICE_TYPE_GPU
                              ? PerplexityCalculator::EDGE_PER_WORKGROUP
                              : PerplexityCalculator::EDGE_PER_THREAD),
                         cfg_, queue_, beta_, pi_.get(), heldoutEdges_,
                         heldoutSet_.get(), compileFlags_, GetBaseFuncs()),
      phiUpdater_((queue_.get_device().type() == CL_DEVICE_TYPE_GPU
                       ? PhiUpdater::NODE_PER_WORKGROUP
                       : PhiUpdater::NODE_PER_THREAD),
                  cfg_, queue_, beta_, pi_.get(), phi_.get(),
                  trainingSet_.get(), compileFlags_, GetBaseFuncs()),
      betaUpdater_((queue_.get_device().type() == CL_DEVICE_TYPE_GPU
                        ? BetaUpdater::EDGE_PER_WORKGROUP
                        : BetaUpdater::EDGE_PER_THREAD),
                   cfg, queue_, theta_, beta_, pi_.get(), trainingSet_.get(),
                   compileFlags_, GetBaseFuncs()) {
  LOG(INFO) << "LEARNER FLAGS = " << compileFlags_;
  // gamma generator
  std::mt19937 mt19937(42);
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  GenerateAndNormalize(&queue_, &gamma, &theta_, &beta_, 2);
  GenerateAndNormalize(&queue_, &gamma, phi_.get(), pi_.get());
}

void Learner::sampleMiniBatch(std::vector<Edge>* edges) {
  std::set<Edge> set;
  while (set.size() < cfg_.mini_batch_size) {
    Vertex u = rand() % cfg_.N;
    auto& neighbors = cfg_.trainingGraph->NeighborsOf(u);
    for (auto& v : neighbors) {
      if (set.size() < cfg_.mini_batch_size) {
        Edge e = MakeEdge(std::min(u, v), std::max(u, v));
        set.insert(e);
      } else {
        break;
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
}

void Learner::run(uint32_t max_iters) {
  uint32_t step_count = 1;
  std::vector<Edge> edges;
  compute::vector<Edge> dev_edges(cfg_.mini_batch_size, queue_.get_context());
  std::set<Vertex> nodes;
  std::vector<Vertex> nodes_vec;
  compute::vector<Vertex> dev_nodes(2 * cfg_.mini_batch_size,
                                    queue_.get_context());
  std::vector<Vertex> neighbors_vec;
  compute::vector<Vertex> dev_neighbors(
      2 * cfg_.mini_batch_size * cfg_.num_node_sample, queue_.get_context());
  uint64_t time = 0;
  for (; step_count < max_iters; ++step_count) {
    auto t1 = std::chrono::high_resolution_clock::now();
    if ((step_count - 1) % cfg_.ppx_interval == 0) {
      Float ppx = heldoutPerplexity_();
      LOG(INFO) << "ppx[" << step_count << "] = " << ppx << ", " << std::exp(ppx);
    }
    edges.clear();
    sampleMiniBatch(&edges);
    nodes.clear();
    for (auto e : edges) {
      Vertex u, v;
      std::tie(u, v) = Vertices(e);
      nodes.insert(u);
      nodes.insert(v);
    }
    nodes_vec.clear();
    nodes_vec.insert(nodes_vec.begin(), nodes.begin(), nodes.end());
    neighbors_vec.resize(nodes.size() * cfg_.num_node_sample);
    std::generate(neighbors_vec.begin(), neighbors_vec.end(),
                  [this]() { return rand() % cfg_.N; });
    compute::copy(edges.begin(), edges.end(), dev_edges.begin(), queue_);
    compute::copy(nodes_vec.begin(), nodes_vec.end(), dev_nodes.begin(),
                  queue_);
    compute::copy(neighbors_vec.begin(), neighbors_vec.end(),
                  dev_neighbors.begin(), queue_);
    phiUpdater_(dev_nodes, dev_neighbors, nodes.size());
    betaUpdater_(&dev_edges, edges.size(), 0.1);
    auto t2 = std::chrono::high_resolution_clock::now();
    time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }
  LOG(INFO) << "TOTAL TIME: " << time / 1e9;
}

}  // namespace mcmc
