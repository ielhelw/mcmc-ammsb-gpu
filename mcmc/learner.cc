#include "mcmc/learner.h"

#include <algorithm>
#include <future>
#include <glog/logging.h>
#include <random>
#include <iomanip>
#include <unordered_set>
#include <queue>

#include "mcmc/algorithm/sum.h"

using namespace std::chrono;

namespace mcmc {

const std::string Learner::GetBaseFuncs() {
  const std::string kSourceBaseFuncs = GetSourceGuard() + GetClTypes() + "\n" +
                                       OpenClSetFactory::GetHeader() + "\n" +
                                       R"%%(
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
  uint64_t num_links =
      static_cast<uint64_t>(cfg.training_ppx_ratio * cfg.training_edges.size());
  uint64_t num_non_links =
      static_cast<uint64_t>(num_links * total / static_cast<double>(cfg.E));
  LOG(INFO) << "TRAINING PPX: LINKS = " << num_links
            << ", NON_LINKS = " << num_non_links;
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
      trainingSet_(setFactory_->CreateSet(*cfg_.training)),
      heldoutSet_(setFactory_->CreateSet(*cfg_.heldout)),
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
          compileFlags_, GetBaseFuncs()),
      stepCount_(1),
      time_(0),
      samplingTime_(0),
#ifdef MCMC_SAMPLE_PARALLEL
      samples_({Sample(cfg_, queue_), Sample(cfg_, queue_)}),
#else
      samples_({Sample(cfg_, queue_)}),
#endif
      phase_(0) {
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

Float Learner::SampleMiniBatch(std::vector<Edge>* edges, unsigned int* seed) {
  return sampler_(cfg_, edges, seed);
}

void Learner::ExtractNodesFromMiniBatch(const std::vector<Edge>& edges,
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
  Float weight = SampleMiniBatch(&sample->edges, &sample->seed);
  ExtractNodesFromMiniBatch(sample->edges, &sample->nodes_vec);
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

Float Learner::HeldoutPerplexity() {
  auto T1 = high_resolution_clock::now();
  Float ppx = heldoutPerplexity_();
  auto T2 = high_resolution_clock::now();
  time_ += duration_cast<nanoseconds>(T2 - T1).count();
  return std::exp(ppx);
}

#ifdef MCMC_CALC_TRAIN_PPX
Float Learner::TrainingPerplexity() {
  auto T1 = high_resolution_clock::now();
  Float train_ppx = trainingPerplexity_();
  auto T2 = high_resolution_clock::now();
  time_ += duration_cast<nanoseconds>(T2 - T1).count();
  return std::exp(train_ppx);
}
#endif

void Learner::Run(uint32_t max_iters, sig_atomic_t* signaled) {
  auto T1 = high_resolution_clock::now();
#ifdef MCMC_SAMPLE_PARALLEL
  if (stepCount_ == 1) {
    futures_[phase_] = std::async(std::launch::async, &Learner::DoSample, this,
                                  &samples_[phase_]);
  }
#endif
  for (uint64_t I = 0;
       I < max_iters && (signaled != nullptr ? !*signaled : true);
       ++I, ++stepCount_) {
    auto tsampling_start = high_resolution_clock::now();
#ifdef MCMC_SAMPLE_PARALLEL
    Float weight = futures_[phase_].get();
    futures_[1 - phase_] = std::async(std::launch::async, &Learner::DoSample,
                                      this, &samples_[1 - phase_]);
#else
    Float weight = DoSample(&samples_[phase_]);
#endif
    auto tsampling_end = high_resolution_clock::now();
    samplingTime_ +=
        duration_cast<nanoseconds>(tsampling_end - tsampling_start).count();

    phiUpdater_(samples_[phase_].dev_nodes,
                samples_[phase_].neighbor_sampler.GetData(),
                samples_[phase_].nodes_vec.size());

    betaUpdater_(&samples_[phase_].dev_edges, samples_[phase_].edges.size(),
                 weight);

#ifdef MCMC_SAMPLE_PARALLEL
    phase_ = 1 - phase_;
#endif
  }
  auto T2 = high_resolution_clock::now();
  time_ += duration_cast<nanoseconds>(T2 - T1).count();
}

void Learner::PrintStats() {
  LOG(INFO) << "TOTAL    : " << time_ / 1.0e9;

  LOG(INFO) << "PPX CALC : " << heldoutPerplexity_.PerplexityTime() / 1.0e3
            << " (%" << 100 * (heldoutPerplexity_.PerplexityTime() / 1.0e3) /
                            (time_ / 1.0e9) << ")";
  LOG(INFO) << "PPX ACCUM: " << heldoutPerplexity_.AccumulateTime() / 1.0e3
            << " (%" << 100 * (heldoutPerplexity_.AccumulateTime() / 1.0e3) /
                            (time_ / 1.0e9) << ")";

#ifdef MCMC_CALC_TRAIN_PPX
  LOG(INFO) << "TRAIN PPX CALC : "
            << trainingPerplexity_.PerplexityTime() / 1.0e3 << " (%"
            << 100 * (trainingPerplexity_.PerplexityTime() / 1.0e3) /
                   (time_ / 1.0e9) << ")";
  LOG(INFO) << "TRAIN PPX ACCUM: "
            << trainingPerplexity_.AccumulateTime() / 1.0e3 << " (%"
            << 100 * (trainingPerplexity_.AccumulateTime() / 1.0e3) /
                   (time_ / 1.0e9) << ")";
#endif
  LOG(INFO) << "SAMPLING : " << samplingTime_ / 1.0e9 << " (%"
            << 100 * (samplingTime_ / 1.0e9) / (time_ / 1.0e9) << ")";

  LOG(INFO) << "PHI      : " << phiUpdater_.UpdatePhiTime() / 1.0e3 << " (%"
            << 100 * (phiUpdater_.UpdatePhiTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
  LOG(INFO) << "PI       : " << phiUpdater_.UpdatePiTime() / 1.0e3 << " (%"
            << 100 * (phiUpdater_.UpdatePiTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";

  LOG(INFO) << "THETA SUM   : " << betaUpdater_.ThetaSumTime() / 1.0e3 << " (%"
            << 100 * (betaUpdater_.ThetaSumTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
  LOG(INFO) << "GRADS PAR   : " << betaUpdater_.GradsPartialTime() / 1.0e3
            << " (%"
            << 100 * (betaUpdater_.GradsPartialTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
  LOG(INFO) << "GRADS SUM   : " << betaUpdater_.GradsSumTime() / 1.0e3 << " (%"
            << 100 * (betaUpdater_.GradsSumTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
  LOG(INFO) << "UPDATE THETA: " << betaUpdater_.UpdateThetaTime() / 1.0e3
            << " (%"
            << 100 * (betaUpdater_.UpdateThetaTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
  LOG(INFO) << "NORM THETA  : " << betaUpdater_.NormalizeTime() / 1.0e3 << " (%"
            << 100 * (betaUpdater_.NormalizeTime() / 1.0e3) / (time_ / 1.0e9)
            << ")";
}

bool Learner::Serialize(std::ostream* out) {
  LearnerProperties props;
  props.set_stepcount(stepCount_);
  props.set_time(time_);
  props.set_samplingtime(samplingTime_);
  props.set_phase(phase_);
#ifdef MCMC_SAMPLE_PARALLEL
  Float weight = futures_[phase_].get();
  // reset future
  futures_[phase_] =
      std::async(std::launch::async, [weight]()->Float { return weight; });
#else
  Float weight = 0;
#endif
  props.set_weight(weight);
  return (::mcmc::Serialize(out, &beta_, &queue_) &&
          ::mcmc::Serialize(out, &theta_, &queue_) &&
          ::mcmc::Serialize(out, pi_.get(), &queue_) &&
          ::mcmc::Serialize(out, &phi_, &queue_) &&
          phiUpdater_.Serialize(out) && betaUpdater_.Serialize(out) &&
#ifdef MCMC_CALC_TRAIN_PPX
          trainingPerplexity_.Serialize(out) &&
#endif
          heldoutPerplexity_.Serialize(out) &&
          ::mcmc::SerializeMessage(out, props)
#ifdef MCMC_SAMPLE_PARALLEL
          && samples_[0].Serialize(out) && samples_[1].Serialize(out)
#endif
          );
}

bool Learner::Parse(std::istream* in) {
  LearnerProperties props;
  if (::mcmc::Parse(in, &beta_, &queue_) &&
      ::mcmc::Parse(in, &theta_, &queue_) &&
      ::mcmc::Parse(in, pi_.get(), &queue_) &&
      ::mcmc::Parse(in, &phi_, &queue_) && phiUpdater_.Parse(in) &&
      betaUpdater_.Parse(in) &&
#ifdef MCMC_CALC_TRAIN_PPX
      trainingPerplexity_.Parse(in) &&
#endif
      heldoutPerplexity_.Parse(in) && ::mcmc::ParseMessage(in, &props)) {
    stepCount_ = props.stepcount();
    time_ = props.time();
    samplingTime_ = props.samplingtime();
    phase_ = props.phase();
#ifdef MCMC_SAMPLE_PARALLEL
    if (samples_[0].Parse(in) && samples_[1].Parse(in)) {
      Float weight = static_cast<Float>(props.weight());
      futures_[phase_] =
          std::async(std::launch::async, [weight]()->Float { return weight; });
      return true;
    }
#else
    return true;
#endif
  }
  return false;
}

}  // namespace mcmc
