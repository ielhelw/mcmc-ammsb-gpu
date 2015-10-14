#include "mcmc/learner.h"

#include <algorithm>
#include <boost/compute/algorithm/reduce.hpp>
#include <future>
#include <glog/logging.h>
#include <random>
#include <unordered_set>

#include "mcmc/algorithm/sum.h"

using namespace std::chrono;

namespace mcmc {

const std::string& Learner::GetBaseFuncs() {
  static const std::string kSourceBaseFuncs =
      GetSourceGuard() + GetClTypes() + "\n" + OpenClSetFactory::GetHeader() +
      "\n" + BOOST_COMPUTE_STRINGIZE_SOURCE(
                 typedef VERTEX_TYPE Vertex; typedef EDGE_TYPE Edge;
                 inline Vertex Vertex0(Edge e) {
                   return (Vertex)((e & 0xffffffff00000000) >> 32);
                 } inline Vertex
                         Vertex1(Edge e) {
                           return (Vertex)((e & 0x00000000ffffffff));
                         } inline Edge MakeEdge(Vertex u,
                                                Vertex v) {
                           return (((Edge)u) << 32) | v;
                         } inline Float Beta(__global Float * g_beta,
                                             uint k) {
                           return g_beta[(k << 1)];
                         } inline Float Theta0(__global Float * g_theta,
                                               uint k) {
                           return g_theta[k << 1];
                         } inline Float Theta1(__global Float * g_theta,
                                               uint k) {
                           return g_theta[(k << 1) + 1];
                         } inline void SetTheta0(__global Float * g_theta,
                                                 uint k, Float v) {
                           g_theta[k << 1] = v;
                         } inline void SetTheta1(__global Float * g_theta,
                                                 uint k, Float v) {
                           g_theta[(k << 1) + 1] = v;
                         } inline __global Float *
                     Pi(__global Float * pi,
                        Vertex u) { return pi + u * K; } inline Float
                         get_eps_t(uint step_count) {
                   return EPS_A * pow(1 + step_count / EPS_B, -EPS_C);
                 });
  return kSourceBaseFuncs;
}

Learner::Learner(const Config& cfg, compute::command_queue queue)
    : cfg_(cfg),
      queue_(queue),
      beta_(2 * cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      allocFactory_(RowPartitionedMatrixFactory<Float>::New(queue_)),
      pi_(allocFactory_->CreateMatrix(cfg_.N, cfg_.K)),
      phi_(cfg_.N, queue_.get_context()),
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
                  cfg_, queue_, beta_, pi_.get(), phi_,
                  trainingSet_.get(), compileFlags_, GetBaseFuncs()),
      betaUpdater_((queue_.get_device().type() == CL_DEVICE_TYPE_GPU
                        ? BetaUpdater::EDGE_PER_WORKGROUP
                        : BetaUpdater::EDGE_PER_THREAD),
                   cfg, queue_, theta_, beta_, pi_.get(), trainingSet_.get(),
                   compileFlags_, GetBaseFuncs()) {
  LOG(INFO) << "LEARNER FLAGS = " << compileFlags_;
  // gamma generator
  std::mt19937 mt19937(6342455113);
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  GenerateAndNormalize(&queue_, &gamma, &theta_, &beta_, 2);
  GenerateAndNormalize(&queue_, &gamma, phi_, pi_.get());
}

void Learner::sampleMiniBatch(std::vector<Edge>* edges, unsigned int* seed) {
  std::unordered_set<Edge> set;
  while (set.size() < cfg_.mini_batch_size) {
    Vertex u = rand_r(seed) % cfg_.N;
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

void Learner::sampleNeighbors(const std::vector<Vertex>& nodes,
                              std::vector<Vertex>* neighbors,
                              std::vector<unsigned int>* seeds) {
  neighbors->resize(nodes.size() * cfg_.num_node_sample);
  std::vector<std::future<void>> futures;
  for (uint32_t i = 0; i < nodes.size(); ++i) {
    futures.push_back(
        std::async(std::launch::async, [this, i, &nodes, neighbors, seeds]() {
          std::unordered_set<Vertex> sample;
          while (sample.size() < cfg_.num_node_sample) {
            Vertex u = rand_r(&((*seeds)[i])) % cfg_.N;
            if (u != nodes[i]) {
              if (sample.insert(u).second) {
                size_t idx = i * cfg_.num_node_sample + sample.size() - 1;
                (*neighbors)[idx] = u;
              }
            }
          }
        }));
  }
  for (uint32_t i = 0; i < nodes.size(); ++i) {
    futures[i].wait();
  }
}

Learner::Sample::Sample(const Config& cfg, compute::command_queue queue)
    : dev_edges(cfg.mini_batch_size, queue.get_context()),
      dev_nodes(2 * cfg.mini_batch_size, queue.get_context()),
      dev_neighbors(2 * cfg.mini_batch_size * cfg.num_node_sample,
                    queue.get_context()),
      seeds(2 * cfg.mini_batch_size) {
  std::generate(seeds.begin(), seeds.end(), rand);
}

void Learner::DoSample(Learner::Sample* sample) {
  sample->edges.clear();
  sampleMiniBatch(&sample->edges, &sample->seeds[0]);
  extractNodesFromMiniBatch(sample->edges, &sample->nodes_vec);
  LOG_IF(FATAL, sample->nodes_vec.size() == 0) << "mini-batch size = 0!";
  LOG_IF(FATAL, sample->nodes_vec.size() > 2*cfg_.mini_batch_size) << "mini-batch too big";
  sampleNeighbors(sample->nodes_vec, &sample->neighbors_vec, &sample->seeds);
}

void Learner::run(uint32_t max_iters) {
  uint32_t step_count = 1;
  uint64_t tppx = 0;
  uint64_t tsampling = 0;
  uint64_t tstaging = 0;
  uint64_t tpi = 0;
  uint64_t tbeta = 0;
  Sample samples[2] = {Sample(cfg_, queue_), Sample(cfg_, queue_)};
  std::future<void> futures[2];
  int phase = 0;
  auto T1 = high_resolution_clock::now();
  futures[phase] =
      std::async(std::launch::async, &Learner::DoSample, this, &samples[phase]);
  for (; step_count < max_iters; ++step_count) {
    if ((step_count - 1) % cfg_.ppx_interval == 0) {
      auto tppx_start = high_resolution_clock::now();
      Float ppx = heldoutPerplexity_();
      auto tppx_end = high_resolution_clock::now();
      tppx += duration_cast<nanoseconds>(tppx_end - tppx_start).count();
      LOG(INFO) << "ppx[" << step_count << "] = " << ppx << ", "
                << std::exp(ppx);
    }
    auto tsampling_start = high_resolution_clock::now();
    futures[phase].wait();
    futures[1 - phase] = std::async(std::launch::async, &Learner::DoSample,
                                    this, &samples[1 - phase]);
    auto tsampling_end = high_resolution_clock::now();
    tsampling +=
        duration_cast<nanoseconds>(tsampling_end - tsampling_start).count();

    auto tstaging_start = high_resolution_clock::now();
    compute::copy(samples[phase].edges.begin(), samples[phase].edges.end(),
                  samples[phase].dev_edges.begin(), queue_);
    compute::copy(samples[phase].nodes_vec.begin(),
                  samples[phase].nodes_vec.end(),
                  samples[phase].dev_nodes.begin(), queue_);
    compute::copy(samples[phase].neighbors_vec.begin(),
                  samples[phase].neighbors_vec.end(),
                  samples[phase].dev_neighbors.begin(), queue_);
    auto tstaging_end = high_resolution_clock::now();
    tstaging +=
        duration_cast<nanoseconds>(tstaging_end - tstaging_start).count();
    auto tpi_start = high_resolution_clock::now();
    phiUpdater_(samples[phase].dev_nodes, samples[phase].dev_neighbors,
                samples[phase].nodes_vec.size());
    auto tpi_end = high_resolution_clock::now();
    tpi += duration_cast<nanoseconds>(tpi_end - tpi_start).count();

    auto tbeta_start = high_resolution_clock::now();
    betaUpdater_(&samples[phase].dev_edges, samples[phase].edges.size(), 0.1);
    auto tbeta_end = high_resolution_clock::now();
    tbeta += duration_cast<nanoseconds>(tbeta_end - tbeta_start).count();
 
    phase = 1 - phase;
  }
  auto T2 = high_resolution_clock::now();
  uint64_t time = duration_cast<nanoseconds>(T2 - T1).count();
  LOG(INFO) << "TOTAL    : " << time / 1e9;
  LOG(INFO) << "PPX      : " << tppx / 1e9 << " (%"
            << 100 * (tppx / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "SAMPLING : " << tsampling / 1e9 << " (%"
            << 100 * (tsampling / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "STAGING  : " << tstaging / 1e9 << " (%"
            << 100 * (tstaging / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "PI       : " << tpi / 1e9 << " (%"
            << 100 * (tpi / 1e9) / (time / 1e9) << ")";
  LOG(INFO) << "BETA     : " << tbeta / 1e9 << " (%"
            << 100 * (tbeta / 1e9) / (time / 1e9) << ")";
}

}  // namespace mcmc
