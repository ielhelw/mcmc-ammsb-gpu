#include "mcmc/learner.h"

#include <algorithm>
#include <future>
#include <glog/logging.h>
#include <random>
#include <unordered_set>
#include <queue>
#include <signal.h>

#include "mcmc/algorithm/sum.h"

using namespace std::chrono;

namespace mcmc {

const std::string& Learner::GetBaseFuncs() {
  static const std::string kSourceBaseFuncs =
      GetSourceGuard() + GetClTypes() + "\n" + OpenClSetFactory::GetHeader() +
      "\n" + R"%%(
          typedef VERTEX_TYPE Vertex; typedef EDGE_TYPE Edge;
          inline Vertex Vertex0(Edge e) {
            return (Vertex)((e & 0xffffffff00000000) >> 32);
          } inline Vertex Vertex1(Edge e) {
            return (Vertex)((e & 0x00000000ffffffff));
          } inline Edge MakeEdge(Vertex u, Vertex v) {
            return (((Edge)u) << 32) | v;
          } inline Float Beta(GLOBAL Float * g_beta, uint k) {
            return g_beta[(k << 1) + 1];
          } inline Float Theta0(GLOBAL Float * g_theta, uint k) {
            return g_theta[k << 1];
          } inline Float Theta1(GLOBAL Float * g_theta, uint k) {
            return g_theta[(k << 1) + 1];
          } inline void SetTheta0(GLOBAL Float * g_theta, uint k, Float v) {
            g_theta[k << 1] = v;
          } inline void SetTheta1(GLOBAL Float * g_theta, uint k, Float v) {
            g_theta[(k << 1) + 1] = v;
          } inline GLOBAL Float *
              Pi(GLOBAL Float * pi, Vertex u) {
                return pi + u * K;
              } inline Float get_eps_t(uint step_count) {
            return EPS_A * pow(1 + step_count / EPS_B, -EPS_C);
          })%%";
  return kSourceBaseFuncs;
}

#ifdef MCMC_CALC_TRAIN_PPX
std::vector<Edge> MakeEdgesForTrainingPerplexity(const Config& cfg) {
  uint64_t total = (cfg.N * (cfg.N - 1)) / 2;
  uint64_t num_links = static_cast<uint64_t>(cfg.training_ppx_ratio *
                                             cfg.training_edges.size());
  uint64_t num_non_links = static_cast<uint64_t>(num_links * total /
                                                static_cast<double>(cfg.E));
  LOG(INFO) << "TRAINING PPX: LINKS = " << num_links << ", NON_LINKS = " << num_non_links;
  std::vector<Edge> ret(num_links + num_non_links);
  // take first half from training edges
  std::copy(cfg.training_edges.begin(), cfg.training_edges.begin() + num_links,
            ret.begin());
  // randomly generate second half (edges not in training or heldout)
  for (uint32_t i = num_links; i < ret.size(); ++i) {
    Vertex u, v;
    Edge e;
    do {
      u = rand() % cfg.N;
      do {
        v = rand() % cfg.N;
      } while (u == v);
      e = MakeEdge(u, v);
    } while (cfg.training->Has(e) || cfg.heldout->Has(e));
    ret[i] = e;
  }
  return ret;
}
#endif

Learner::Learner(const Config& cfg, clcuda::Queue queue)
    : cfg_(cfg),
      queue_(queue),
      beta_(queue_.GetContext(), 2 * cfg_.K),
      theta_(queue_.GetContext(), 2 * cfg_.K),
      allocFactory_(RowPartitionedMatrixFactory<Float>::New(queue_)),
      pi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
      phi_(queue_.GetContext(), cfg_.N),
      setFactory_(OpenClSetFactory::New(queue_)),
      trainingSet_(setFactory_->CreateSet(cfg_.training->Serialize())),
      heldoutSet_(setFactory_->CreateSet(cfg_.heldout->Serialize())),
      trainingEdges_(queue_.GetContext(), queue_, cfg_.training_edges.begin(),
                     cfg_.training_edges.end()),
      heldoutEdges_(queue_.GetContext(), queue_, cfg_.heldout_edges.begin(),
                    cfg_.heldout_edges.end()),
      compileFlags_(MakeCompileFlags(cfg_)),
#ifdef MCMC_CALC_TRAIN_PPX
      trainingPerplexityEdges_(MakeEdgesForTrainingPerplexity(cfg)),
      devTrainingPerplexityEdges_(queue.GetContext(), queue,
                                  trainingPerplexityEdges_.begin(),
                                  trainingPerplexityEdges_.end()),
      trainingPerplexity_((queue_.GetDevice().Type() == "GPU"
                               ? PerplexityCalculator::EDGE_PER_WORKGROUP
                               : PerplexityCalculator::EDGE_PER_THREAD),
                          cfg_, queue_, beta_, pi_.get(),
                          devTrainingPerplexityEdges_, trainingSet_.get(),
                          compileFlags_, GetBaseFuncs()),
#endif
      heldoutPerplexity_((queue_.GetDevice().Type() == "GPU"
                              ? PerplexityCalculator::EDGE_PER_WORKGROUP
                              : PerplexityCalculator::EDGE_PER_THREAD),
                         cfg_, queue_, beta_, pi_.get(), heldoutEdges_,
                         heldoutSet_.get(), compileFlags_, GetBaseFuncs()),
      phiUpdater_(
          (queue_.GetDevice().Type() == "GPU" ? PhiUpdater::NODE_PER_WORKGROUP
                                              : PhiUpdater::NODE_PER_THREAD),
          cfg_, queue_, beta_, pi_.get(), phi_, trainingSet_.get(),
          compileFlags_, GetBaseFuncs()),
      betaUpdater_(
          (queue_.GetDevice().Type() == "GPU" ? BetaUpdater::EDGE_PER_WORKGROUP
                                              : BetaUpdater::EDGE_PER_THREAD),
          cfg, queue_, theta_, beta_, pi_.get(), trainingSet_.get(),
          compileFlags_, GetBaseFuncs()) {
  switch (cfg_.strategy) {
    case NodeLink:
      sampler_ = sampleNodeLink;
      break;
    case NodeNonLink:
      sampler_ = sampleNodeNonLink;
      break;
    case Node:
      sampler_ = sampleNode;
      break;
    case BFLink:
      sampler_ = sampleBreadthFirstLink;
      break;
    case BFNonLink:
      sampler_ = sampleBreadthFirstNonLink;
      break;
    case BF:
      sampler_ = sampleBreadthFirst;
      break;
    default:
      LOG(FATAL) << "Unkown sample strategy";
  }
  //  LOG(INFO) << "LEARNER FLAGS = " << compileFlags_;
  // gamma generator
  std::mt19937 mt19937(6342455113);
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  random::RandomAndNormalize(&queue_, &gamma, &theta_, &beta_, 2);
  random::RandomGammaAndNormalize(&queue_, cfg_.eta0, cfg_.eta1, pi_.get(),
                                  &phi_);
}

Float Learner::sampleMiniBatch(std::vector<Edge>* edges, unsigned int* seed) {
  return sampler_(cfg_, edges, seed);
}

void Learner::extractNodesFromMiniBatch(const std::vector<Edge>& edges,
                                        std::vector<Vertex>* nodes_vec) {
  std::unordered_set<Vertex> nodes;
  for (auto e : edges) {
    Vertex u, v;
    std::tie(u, v) = Vertices(e);
    nodes.insert(u);
    nodes.insert(v);
  }
  nodes_vec->clear();
  nodes_vec->insert(nodes_vec->begin(), nodes.begin(), nodes.end());
}

Float Learner::DoSample(Sample* sample) {
  sample->edges.clear();
  Float weight = sampleMiniBatch(&sample->edges, &sample->seed);
  extractNodesFromMiniBatch(sample->edges, &sample->nodes_vec);
  LOG_IF(FATAL, sample->nodes_vec.size() == 0) << "mini-batch size = 0!";
  LOG_IF(FATAL, sample->nodes_vec.size() > 2 * cfg_.mini_batch_size)
      << "mini-batch too big";
#ifndef MCMC_USE_CL
  // CUDA NEEDS TO KNOW CURRENT CONTEXT PER THREAD
  cuCtxSetCurrent(sample->queue.GetContext()());
#endif
  sample->dev_edges.Write(sample->queue, sample->edges.size(),
                          sample->edges.data());
  sample->dev_nodes.Write(sample->queue, sample->nodes_vec.size(),
                          sample->nodes_vec.data());
  sample->neighbor_sampler(sample->nodes_vec.size(), &(sample->dev_nodes));
  return weight;
}

sig_atomic_t signaled = 0;

void handler(int sig __attribute((unused))) { signaled = 1; }

void Learner::run(uint32_t max_iters) {
  uint32_t step_count = 1;
  uint64_t tppx = 0;
#ifdef MCMC_CALC_TRAIN_PPX
  uint64_t ttppx = 0;
#endif
  uint64_t tsampling = 0;
  uint64_t tpi = 0;
  uint64_t tbeta = 0;
  Sample samples[2] = {Sample(cfg_, queue_), Sample(cfg_, queue_)};
  std::future<Float> futures[2];
  int phase = 0;
  auto T1 = high_resolution_clock::now();
  futures[phase] =
      std::async(std::launch::async, &Learner::DoSample, this, &samples[phase]);
  auto prev_handler = signal(SIGINT, handler);
  for (; step_count < max_iters && !signaled; ++step_count) {
    if ((step_count - 1) % cfg_.ppx_interval == 0) {
      auto tppx_start = high_resolution_clock::now();
      Float ppx = heldoutPerplexity_();
      auto tppx_end = high_resolution_clock::now();
      tppx += duration_cast<nanoseconds>(tppx_end - tppx_start).count();
      LOG(INFO) << "ppx[" << step_count << "] = " << ppx << ", "
                << std::exp(ppx);
#ifdef MCMC_CALC_TRAIN_PPX
      auto ttppx_start = high_resolution_clock::now();
      Float train_ppx = trainingPerplexity_();
      auto ttppx_end = high_resolution_clock::now();
      ttppx += duration_cast<nanoseconds>(ttppx_end - ttppx_start).count();
      LOG(INFO) << "train-ppx[" << step_count << "] = " << train_ppx << ", "
                << std::exp(train_ppx);
#endif
    }
    auto tsampling_start = high_resolution_clock::now();
    Float weight = futures[phase].get();
    futures[1 - phase] = std::async(std::launch::async, &Learner::DoSample,
                                    this, &samples[1 - phase]);
    auto tsampling_end = high_resolution_clock::now();
    tsampling +=
        duration_cast<nanoseconds>(tsampling_end - tsampling_start).count();

    auto tpi_start = high_resolution_clock::now();
    phiUpdater_(samples[phase].dev_nodes,
                samples[phase].neighbor_sampler.GetData(),
                samples[phase].nodes_vec.size());
    auto tpi_end = high_resolution_clock::now();
    tpi += duration_cast<nanoseconds>(tpi_end - tpi_start).count();

    auto tbeta_start = high_resolution_clock::now();
    betaUpdater_(&samples[phase].dev_edges, samples[phase].edges.size(),
                 weight);
    auto tbeta_end = high_resolution_clock::now();
    tbeta += duration_cast<nanoseconds>(tbeta_end - tbeta_start).count();

    phase = 1 - phase;
  }
  auto T2 = high_resolution_clock::now();
  uint64_t time = duration_cast<nanoseconds>(T2 - T1).count();
  LOG_IF(INFO, signaled) << "FORCED TERMINATE";
  LOG(INFO) << "TOTAL    : " << time / 1e9;
  LOG(INFO) << "PPX      : " << tppx / 1e9 << " (%"
            << 100 * (tppx / 1e9) / (time / 1e9) << ")";
#ifdef MCMC_CALC_TRAIN_PPX
  LOG(INFO) << "TRAIN PPX: " << ttppx / 1e9 << " (%"
            << 100 * (ttppx / 1e9) / (time / 1e9) << ")";
#endif
  LOG(INFO) << "SAMPLING : " << tsampling / 1e9 << " (%"
            << 100 * (tsampling / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "PI       : " << tpi / 1e9 << " (%"
            << 100 * (tpi / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "BETA     : " << tbeta / 1e9 << " (%"
            << 100 * (tbeta / 1e9) / (time / 1e9) << ")";

  signal(SIGINT, prev_handler);
}

}  // namespace mcmc
