#include "mcmc/beta.h"

#include <chrono>
#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string kSourceBetaBase = random::GetRandomHeader() + R"%%(
        KERNEL void sum_theta(GLOBAL Float* g_theta,     // [K, 2]
                              GLOBAL Float* g_theta_sum  // [K]
                              ) {
          uint gsize = GET_GLOBAL_SIZE();
          for (uint i = GET_GLOBAL_ID(); i < K; i += gsize) {
            g_theta_sum[i] = Theta0(g_theta, i) + Theta1(g_theta, i);
          }
        } KERNEL void sum_grads(GLOBAL Float* grads, uint num_partial_sums) {
          uint i = GET_GLOBAL_ID();
          uint gsize = GET_GLOBAL_SIZE();
          for (; i < 2 * K; i += gsize) {
            for (uint p = 1; p < num_partial_sums; ++p) {
              grads[i] += grads[i + p * 2 * K];
            }
          }
        }

        KERNEL void update_theta(GLOBAL Float* theta, GLOBAL Float* grads,
                                 uint step_count, Float scale,
                                 GLOBAL void* vrand) {
          const uint gid = GET_GLOBAL_ID();
          const uint gsize = GET_GLOBAL_SIZE();
          GLOBAL Random* random = (GLOBAL Random*)vrand;
          if (gid < K) {
            random_seed_t rseed = random->base_[gid];
            Float eps_t = get_eps_t(step_count);
            for (uint k = gid; k < K; k += gsize) {
              Float r0 = randn(&rseed);
              Float grads_k = grads[2 * k];
              Float theta_k = Theta0(theta, k);
              Float f0 = SQRT(eps_t * theta_k);
              SetTheta0(theta, k,
                        FABS(theta_k +
                             eps_t / ((Float)2.0) * (ETA0 - theta_k + scale * grads_k) +
                             f0 * r0));
              Float r1 = randn(&rseed);
              Float grads_2k = grads[2 * k + 1];
              Float theta_2k = Theta1(theta, k);
              Float f1 = SQRT(eps_t * theta_2k);
              SetTheta1(theta, k, FABS(theta_2k +
                                       eps_t / ((Float)2.0) * (ETA1 - theta_2k +
                                                      scale * grads_2k) +
                                       f1 * r1));
            }
            random->base_[gid] = rseed;
          }
        }

        )%%";

const std::string kSourceBeta = kSourceBetaBase + R"%%(
    KERNEL void calculate_grads_partial(
        GLOBAL Float* theta,      // [K, 2]
        GLOBAL Float* theta_sum,  // [K]
        GLOBAL Float* beta,       // [K]
        GLOBAL void* g_pi,        // [N, K]
        GLOBAL void* vset /*EdgeSet*/, GLOBAL Edge* mini_batch_edges,
        uint num_mini_batch_edges,
        GLOBAL Float* probs,  // [num_mini_batch_edges * K]
        GLOBAL Float* grads   // min(#edges, #num_threads) * [K, 2]
        ) {
      uint i = GET_GLOBAL_ID();
      uint gsize = GET_GLOBAL_SIZE();
      probs += i * K;
      grads += i * 2 * K;
      // reset grads
      if (i < num_mini_batch_edges) {
        for (uint j = 0; j < 2 * K; ++j) grads[j] = 0;
      }
      for (; i < num_mini_batch_edges; i += gsize) {
        Edge edge = mini_batch_edges[i];
        Vertex u = Vertex0(edge);
        Vertex v = Vertex1(edge);
        edge = MakeEdge(min(u, v), max(u, v));
        uint y = Set_HasEdge((GLOBAL Set*)vset, edge) ? 1 : 0;
        GLOBAL Float* pi_a = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, u);
        GLOBAL Float* pi_b = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, v);
        Float pi_sum = 0;
        Float probs_sum = 0;

        for (uint k = 0; k < K; ++k) {
          Float f = pi_a[k] * pi_b[k];
          pi_sum += f;
          Float probs_k;
          if (y) {
            probs_k = Beta(beta, k) * f;
          } else {
            probs_k = (((Float)1.0) - Beta(beta, k)) * f;
          }
          probs[k] = probs_k;
          probs_sum += probs_k;
        }
        Float prob_0 = (y ? EPSILON : (((Float)1.0) - EPSILON)) * (((Float)1.0) - pi_sum);
        probs_sum += prob_0;
        for (uint k = 0; k < K; k++) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = ((Float)1.0) / theta_sum[k];
          grads[2 * k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);
          grads[2 * k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);
        }
      }
    })%%";

const std::string kSourceBetaWg =
    mcmc::algorithm::WorkGroupSum(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() + "\n" +
    kSourceBetaBase + R"%%(
    KERNEL void calculate_grads_partial(
        GLOBAL Float* theta,      // [K, 2]
        GLOBAL Float* theta_sum,  // [K]
        GLOBAL Float* beta,       // [K]
        GLOBAL void* g_pi,        // [N, K]
        GLOBAL void* vset /*EdgeSet*/, GLOBAL Edge* mini_batch_edges,
        uint num_mini_batch_edges,
        GLOBAL Float* probs,    // [num_mini_batch_edges * K]
        GLOBAL Float* grads,    // min(#edges, #num_threads) * [K, 2]
        GLOBAL Float* scratch  // min(#edges, #num_threads) * [K]
      ) {
      LOCAL_DECLARE Float aux[WG_SIZE];
      uint i = GET_GROUP_ID();
      const uint gsize = GET_NUM_GROUPS();
      const uint lid = GET_LOCAL_ID();
      const uint lsize = GET_LOCAL_SIZE();
      probs += i * K;
      grads += i * 2 * K;
      scratch += i * K;
      // reset grads
      if (i < num_mini_batch_edges) {
        for (uint j = lid; j < 2 * K; j += lsize) grads[j] = 0;
      }
      for (; i < num_mini_batch_edges; i += gsize) {
        Edge edge = mini_batch_edges[i];
        Vertex u = Vertex0(edge);
        Vertex v = Vertex1(edge);
        edge = MakeEdge(min(u, v), max(u, v));
        uint y = Set_HasEdge((GLOBAL Set*)vset, edge) ? 1 : 0;
        GLOBAL Float* pi_a = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, u);
        GLOBAL Float* pi_b = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, v);
        Float pi_sum = 0;
        Float probs_sum = 0;

        for (uint k = lid; k < K; k += lsize) {
          Float f = pi_a[k] * pi_b[k];
          scratch[k] = f;
          Float probs_k;
          Float beta_k = Beta(beta, k);
          if (y) {
            probs_k = beta_k * f;
          } else {
            probs_k = (((Float)1.0) - beta_k) * f;
          }
          probs[k] = probs_k;
        }
        BARRIER_GLOBAL;
        WG_SUM_Float(scratch, aux, K);
        pi_sum = aux[0];
        BARRIER_GLOBAL;
        WG_SUM_Float(probs, aux, K);
        probs_sum = aux[0];

        Float prob_0 = (y ? EPSILON : (((Float)1.0) - EPSILON)) * (((Float)1.0) - pi_sum);
        probs_sum += prob_0;
        for (uint k = lid; k < K; k += lsize) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = ((Float)1.0) / theta_sum[k];
          grads[2 * k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);
          grads[2 * k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);
        }
      }
    })%%";

BetaUpdater::BetaUpdater(Mode mode, const Config& cfg, clcuda::Queue queue,
                         clcuda::Buffer<Float>& theta,
                         clcuda::Buffer<Float>& beta,
                         RowPartitionedMatrix<Float>* pi,
                         OpenClSet* trainingSet,
                         const std::vector<std::string>& compileFlags,
                         const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      theta_(theta),
      beta_(beta),
      pi_(pi),
      trainingSet_(trainingSet),
      normalizer_(queue_, &beta_, 2, 1),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(
          cfg.K, random::random_seed_t{cfg.beta_seed[0], cfg.beta_seed[1]})),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.beta_wg_size),
      theta_sum_(queue_.GetContext(), cfg.K),
      grads_(queue_.GetContext(), cfg.mini_batch_size * 2 * cfg.K),
      probs_(queue_.GetContext(), cfg.mini_batch_size * cfg.K) {
  const std::string* src = nullptr;
  switch (mode) {
    case EDGE_PER_THREAD:
      src = &kSourceBeta;
      break;
    case EDGE_PER_WORKGROUP:
      scratch_.reset(new clcuda::Buffer<Float>(
          queue_.GetContext(), cfg.mini_batch_size * 2 * cfg.K));
      src = &kSourceBetaWg;
      break;
    default:
      LOG(FATAL) << "Failed to recognize mode";
  }
  prog_.reset(new clcuda::Program(
      queue_.GetContext(),
      baseFuncs + GetRowPartitionedMatrixHeader<Float>() +
          "#define FloatRowPartitionedMatrix " + type_name<Float>() +
          "RowPartitionedMatrix\n"
          "#define FloatRowPartitionedMatrix_Row " +
          type_name<Float>() + "RowPartitionedMatrix_Row\n" + *src));
  std::vector<std::string> opts =
      ::mcmc::GetClFlags(mode_ == EDGE_PER_WORKGROUP ? local_ : 0);
  opts.insert(opts.end(), compileFlags.begin(), compileFlags.end());
  clcuda::BuildStatus status = prog_->Build(queue_.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_->GetBuildInfo(queue_.GetDevice());
  LOG(INFO) << "####################### BETA LOG:" << std::endl
            << prog_->GetBuildInfo(queue_.GetDevice());
  theta_sum_kernel_.reset(new clcuda::Kernel(*prog_, "sum_theta"));
  theta_sum_kernel_->SetArgument(0, theta_);
  theta_sum_kernel_->SetArgument(1, theta_sum_);

  grads_partial_kernel_.reset(
      new clcuda::Kernel(*prog_, "calculate_grads_partial"));
  grads_partial_kernel_->SetArgument(0, theta_);
  grads_partial_kernel_->SetArgument(1, theta_sum_);
  grads_partial_kernel_->SetArgument(2, beta_);
  grads_partial_kernel_->SetArgument(3, pi_->Get());
  grads_partial_kernel_->SetArgument(4, trainingSet->Get()());
  grads_partial_kernel_->SetArgument(7, probs_);
  grads_partial_kernel_->SetArgument(8, grads_);
  if (mode_ == EDGE_PER_WORKGROUP) {
    grads_partial_kernel_->SetArgument(9, *scratch_);
  }

  grads_sum_kernel_.reset(new clcuda::Kernel(*prog_, "sum_grads"));
  grads_sum_kernel_->SetArgument(0, grads_);

  update_theta_kernel_.reset(new clcuda::Kernel(*prog_, "update_theta"));
  update_theta_kernel_->SetArgument(0, theta_);
  update_theta_kernel_->SetArgument(1, grads_);
  update_theta_kernel_->SetArgument(4, rand_->Get());
}

void BetaUpdater::operator()(clcuda::Buffer<Edge>* edges, uint32_t num_edges,
                             Float scale) {
  ++count_calls_;
  {
    theta_sum_kernel_->Launch(queue_, {k_}, {32}, theta_sum_event_);
    queue_.Finish();
  }
  {
    uint32_t global = 0;
    grads_partial_kernel_->SetArgument(5, *edges);
    grads_partial_kernel_->SetArgument(6, num_edges);
    if (mode_ == EDGE_PER_THREAD) {
      global = (num_edges / local_ + (num_edges % local_ ? 1 : 0));
    } else {
      global = num_edges;
    }
    global = std::min(global, GetMaxGroups()) * local_;
    grads_partial_kernel_->Launch(queue_, {global}, {local_},
                                  grads_partial_event_);
    queue_.Finish();

    uint32_t num_partials = std::min(global, num_edges);
    grads_sum_kernel_->SetArgument(1, num_partials);
    grads_sum_kernel_->Launch(queue_, {2 * k_}, {32}, grads_sum_event_);
    queue_.Finish();
  }
  {
    update_theta_kernel_->SetArgument(2, count_calls_);
    update_theta_kernel_->SetArgument(3, scale);
    update_theta_kernel_->Launch(queue_, {k_}, {32}, update_theta_event_);
    queue_.Finish();
  }
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    theta_.CopyTo(queue_, theta_.GetSize() / sizeof(Float), beta_);
    normalizer_();
    auto t2 = std::chrono::high_resolution_clock::now();
    normalize_time_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t2).count();
  }
}

uint64_t BetaUpdater::LastInvocationTime() const {
  return theta_sum_event_.GetElapsedTime() +
         grads_partial_event_.GetElapsedTime() +
         grads_sum_event_.GetElapsedTime() +
         update_theta_event_.GetElapsedTime() + normalize_time_;
}

bool BetaUpdater::Serialize(std::ostream* out) {
  BetaProperties props;
  props.set_count_calls(count_calls_);
  return (rand_->Serialize(out) &&
          ::mcmc::Serialize(out, &theta_sum_, &queue_) &&
          ::mcmc::SerializeMessage(out, props));
}

bool BetaUpdater::Parse(std::istream* in) {
  BetaProperties props;
  if (rand_->Parse(in) && ::mcmc::Parse(in, &theta_sum_, &queue_) &&
      ::mcmc::ParseMessage(in, &props)) {
    count_calls_ = props.count_calls();
    return true;
  } else {
    return false;
  }
}

}  // namespace mcmc
