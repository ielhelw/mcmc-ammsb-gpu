#include "mcmc/beta.h"

#include <boost/algorithm/string/replace.hpp>
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
            Float sum = grads[i];
            for (uint p = 1; p < num_partial_sums; ++p) {
              sum += grads[i + p * 2 * K];
            }
            grads[i] = sum;
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
                             eps_t / FL(2.0) * (ETA0 - theta_k + scale * grads_k) +
                             f0 * r0));
              Float r1 = randn(&rseed);
              Float grads_2k = grads[2 * k + 1];
              Float theta_2k = Theta1(theta, k);
              Float f1 = SQRT(eps_t * theta_2k);
              SetTheta1(theta, k, FABS(theta_2k +
                                       eps_t / FL(2.0) * (ETA1 - theta_2k +
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
            probs_k = (FL(1.0) - Beta(beta, k)) * f;
          }
          probs[k] = probs_k;
          probs_sum += probs_k;
        }
        Float prob_0 = (y ? EPSILON : (FL(1.0) - EPSILON)) * (FL(1.0) - pi_sum);
        probs_sum += prob_0;
        for (uint k = 0; k < K; k++) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = FL(1.0) / theta_sum[k];
          grads[2 * k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);
          grads[2 * k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);
        }
      }
    })%%";

const std::string kSourceBetaWg =
    mcmc::algorithm::WorkGroupSum(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() + "\n" +
    "#define WG_SUM_Float_LOCAL_ WG_SUM_" + type_name<Float>() + "_LOCAL_\n" +
    kSourceBetaBase + R"%%(
    #define K_PER_THREAD ((K/WG_SIZE) + (K % WG_SIZE? 1 : 0))
    #define CALC_PROBS(i, kk) \
    {\
      uint k = kk; \
      if (k < K) { \
          Float f = pi_a[k] * pi_b[k]; \
          scratch += f; \
          Float probs_k; \
          Float beta_k = Beta(beta, k); \
          if (y) { \
            probs_k = beta_k * f; \
          } else { \
            probs_k = (FL(1.0) - beta_k) * f; \
          } \
          probs[i] = probs_k; \
          probs_sum += probs_k; \
      } \
    }
    #define CALC_GRADS(i, kk) \
    {\
      uint k = kk; \
      if (k < K) {\
        Float f = probs[i] / probs_sum;\
        Float one_over_theta_sum = FL(1.0) / theta_sum[k];\
        lgrads[2 * k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);\
        lgrads[2 * k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);\
      }\
    }


    KERNEL void calculate_grads_partial(
        GLOBAL Float* theta,      // [K, 2]
        GLOBAL Float* theta_sum,  // [K]
        GLOBAL Float* beta,       // [K]
        GLOBAL void* g_pi,        // [N, K]
        GLOBAL void* vset /*EdgeSet*/, GLOBAL Edge* mini_batch_edges,
        uint num_mini_batch_edges,
        GLOBAL Float* grads     // min(#edges, #num_threads) * [K, 2]
      ) {
      LOCAL_DECLARE Float aux[WG_SIZE];
      LOCAL_DECLARE Float lgrads[2 * K];
      Float probs[K_PER_THREAD];
      uint i = GET_GROUP_ID();
      const uint gsize = GET_NUM_GROUPS();
      const uint lid = GET_LOCAL_ID();
      grads += i * 2 * K;
      // reset grads
      if (i < num_mini_batch_edges) {
        for (uint j = lid; j < 2 * K; j += WG_SIZE) lgrads[j] = 0;
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
        Float scratch = 0;
        {
          GENERATE_CALC_PROBS
          // CALC_PROBS(0, lid + 0 * WG_SIZE);
        }
        aux[lid] = scratch;
        BARRIER_LOCAL;
        WG_SUM_Float_LOCAL_(aux, K);
        pi_sum = aux[0];
        BARRIER_LOCAL;
        aux[lid] = probs_sum;
        BARRIER_LOCAL;
        WG_SUM_Float_LOCAL_(aux, K);
        probs_sum = aux[0];

        Float prob_0 = (y ? EPSILON : (FL(1.0) - EPSILON)) * (FL(1.0) - pi_sum);
        probs_sum += prob_0;
        {
          GENERATE_CALC_GRADS
          //  CALC_GRADS(0, lid + 0 * WG_SIZE);
        }

      }
      if (GET_GROUP_ID() < num_mini_batch_edges) {
        BARRIER_LOCAL;
        for (uint k = lid; k < K; k += WG_SIZE) {
          grads[2 * k    ] = lgrads[2 * k    ];
          grads[2 * k + 1] = lgrads[2 * k + 1];
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
      t_theta_sum_(0),
      t_grads_partial_(0),
      t_grads_sum_(0),
      t_update_theta_(0),
      t_normalize_(0) {
  std::string src;
  switch (mode) {
    case EDGE_PER_THREAD:
      src = kSourceBeta;
      break;
    case EDGE_PER_WORKGROUP: {
      src = kSourceBetaWg;
      uint32_t k_per_thread = k_ / local_ + (k_ % local_ ? 1 : 0);
      for (auto s : std::vector<std::string>{"CALC_PROBS", "CALC_GRADS"}) {
        std::ostringstream out;
        for (uint i = 0; i < k_per_thread; ++i) {
          // CALC_PROBS(0, lid + 0 * WG_SIZE);
          out << s << "(" << i << ", lid + " << i << " * WG_SIZE);\n";
        }
        src = boost::replace_all_copy(src, std::string("GENERATE_") + s,
                                      out.str());
      }
    } break;
    default:
      LOG(FATAL) << "Failed to recognize mode";
  }
  prog_.reset(new clcuda::Program(
      queue_.GetContext(),
      baseFuncs + GetRowPartitionedMatrixHeader<Float>() +
          "#define FloatRowPartitionedMatrix " + type_name<Float>() +
          "RowPartitionedMatrix\n"
          "#define FloatRowPartitionedMatrix_Row " +
          type_name<Float>() + "RowPartitionedMatrix_Row\n" + src));
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
  if (mode == EDGE_PER_THREAD) {
    probs_.reset(new clcuda::Buffer<Float>(queue_.GetContext(),
                                           cfg.mini_batch_size * cfg.K));
    grads_partial_kernel_->SetArgument(7, *probs_);
    grads_partial_kernel_->SetArgument(8, grads_);
  } else {
    grads_partial_kernel_->SetArgument(7, grads_);
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
    clcuda::Event theta_sum_event;
    theta_sum_kernel_->Launch(queue_, {k_}, {32}, theta_sum_event);
    queue_.Finish();
    t_theta_sum_ += theta_sum_event.GetElapsedTime();
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
    clcuda::Event grads_partial_event;
    grads_partial_kernel_->Launch(queue_, {global}, {local_},
                                  grads_partial_event);
    queue_.Finish();
    t_grads_partial_ += grads_partial_event.GetElapsedTime();

    uint32_t num_partials = std::min(global, num_edges);
    grads_sum_kernel_->SetArgument(1, num_partials);
    clcuda::Event grads_sum_event;
    grads_sum_kernel_->Launch(queue_, {2 * k_}, {32}, grads_sum_event);
    queue_.Finish();
    t_grads_sum_ += grads_sum_event.GetElapsedTime();
  }
  {
    update_theta_kernel_->SetArgument(2, count_calls_);
    update_theta_kernel_->SetArgument(3, scale);
    clcuda::Event update_theta_event;
    update_theta_kernel_->Launch(queue_, {k_}, {32}, update_theta_event);
    queue_.Finish();
    t_update_theta_ += update_theta_event.GetElapsedTime();
  }
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    theta_.CopyTo(queue_, theta_.GetSize() / sizeof(Float), beta_);
    normalizer_();
    auto t2 = std::chrono::high_resolution_clock::now();
    t_normalize_ +=
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();
  }
}

bool BetaUpdater::Serialize(std::ostream* out) {
  BetaProperties props;
  props.set_count_calls(count_calls_);
  props.set_theta_sum_time(t_theta_sum_);
  props.set_grads_partial_time(t_grads_partial_);
  props.set_grads_sum_time(t_grads_sum_);
  props.set_update_theta_time(t_update_theta_);
  props.set_normalize_time(t_normalize_);
  return (rand_->Serialize(out) &&
          ::mcmc::Serialize(out, &theta_sum_, &queue_) &&
          ::mcmc::SerializeMessage(out, props));
}

bool BetaUpdater::Parse(std::istream* in) {
  BetaProperties props;
  if (rand_->Parse(in) && ::mcmc::Parse(in, &theta_sum_, &queue_) &&
      ::mcmc::ParseMessage(in, &props)) {
    count_calls_ = props.count_calls();
    t_theta_sum_ = props.theta_sum_time();
    t_grads_partial_ = props.grads_partial_time();
    t_grads_sum_ = props.grads_sum_time();
    t_update_theta_ = props.update_theta_time();
    t_normalize_ = props.normalize_time();
    return true;
  } else {
    return false;
  }
}

}  // namespace mcmc
