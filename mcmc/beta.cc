#include "mcmc/beta.h"

#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string kSourceBeta = BOOST_COMPUTE_STRINGIZE_SOURCE(
    __kernel void sum_theta(__global Float* g_theta,     // [K, 2]
                            __global Float* g_theta_sum  // [K]
                            ) {
      uint gsize = get_global_size(0);
      for (uint i = get_global_id(0); i < K; i += gsize) {
        g_theta_sum[i] = g_theta[i] + g_theta[K + i];
      }
    }

    __kernel void calculate_grads_partial(
        __global Float* theta,      // [K, 2]
        __global Float* theta_sum,  // [K]
        __global Float* beta,       // [K]
        __global Float* g_pi,       // [N, K]
        __global void* vset /*EdgeSet*/, __global Edge* mini_batch_edges,
        uint num_mini_batch_edges,
        __global Float* probs,  // [num_mini_batch_edges * K]
        __global Float* grads   // min(#edges, #num_threads) * [K, 2]
        ) {
      uint i = get_global_id(0);
      uint gsize = get_global_size(0);
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
        uint y = Set_HasEdge(vset, edge) ? 1 : 0;
        __global Float* pi_a = Pi(g_pi, u);
        __global Float* pi_b = Pi(g_pi, v);
        Float pi_sum = 0;
        Float probs_sum = 0;

        for (uint k = 0; k < K; ++k) {
          Float f = pi_a[k] * pi_b[k];
          pi_sum += f;
          Float probs_k;
          if (y) {
            probs_k = beta[k] * f;
          } else {
            probs_k = (1.0 - beta[k]) * f;
          }
          probs[k] = probs_k;
          probs_sum += probs_k;
        }
        Float prob_0 = (y ? EPSILON : (1.0 - EPSILON)) * (1.0 - pi_sum);
        probs_sum += prob_0;
        for (uint k = 0; k < K; k++) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = 1.0 / theta_sum[k];
          grads[k] += f * ((1 - y) / theta[k] - one_over_theta_sum);
          grads[k + K] += f * (y / theta[k + K] - one_over_theta_sum);
        }
      }
    }

    __kernel void sum_grads(__global Float* grads, uint num_partial_sums) {
      uint i = get_global_id(0);
      uint gsize = get_global_size(0);
      for (; i < 2 * K; i += gsize) {
        for (uint p = 1; p < num_partial_sums; ++p) {
          grads[i] += grads[i + p * 2 * K];
        }
      }
    }

    __kernel void update_theta(__global Float* theta, __global Float* grads,
                               uint step_count, Float scale) {
      uint gsize = get_global_size(0);
      Float eps_t = get_eps_t(step_count);
      for (uint k = get_global_id(0); k < K; k += gsize) {
        Float f0 = sqrt(eps_t * theta[k]);
        Float f1 = sqrt(eps_t * theta[k + K]);
        theta[k] =
            fabs(theta[k] + eps_t / 2.0 * (ETA0 - theta[k] + scale * grads[k]) +
                 f0 * 1 /*randn noise*/);
        theta[k + K] =
            fabs(theta[k + K] +
                 eps_t / 2.0 * (ETA1 - theta[k + K] + scale * grads[k + K]) +
                 f0 * 1 /*randn noise*/);
      }
    }

    );

const std::string kSourceBetaWg =
    mcmc::algorithm::WorkGroupSum(compute::type_name<Float>()) + "\n" +
    "#define WG_SUM_FOLD_Float WG_SUM_FOLD_" + compute::type_name<Float>() +
    "\n"
    "#define WG_SUM_Float WG_SUM_" +
    compute::type_name<Float>() + "\n" +
    BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void sum_theta(__global Float* g_theta,     // [K, 2]
                                __global Float* g_theta_sum  // [K]
                                ) {
          uint gsize = get_global_size(0);
          for (uint i = get_global_id(0); i < K; i += gsize) {
            g_theta_sum[i] = g_theta[i] + g_theta[K + i];
          }
        }

        __kernel void calculate_grads_partial(
            __global Float* theta,      // [K, 2]
            __global Float* theta_sum,  // [K]
            __global Float* beta,       // [K]
            __global Float* g_pi,       // [N, K]
            __global void* vset /*EdgeSet*/, __global Edge* mini_batch_edges,
            uint num_mini_batch_edges,
            __global Float* probs,    // [num_mini_batch_edges * K]
            __global Float* grads,    // min(#edges, #num_threads) * [K, 2]
            __global Float* scratch,  // min(#edges, #num_threads) * [K]
            __local Float* aux) {
          uint i = get_group_id(0);
          const uint gsize = get_num_groups(0);
          const uint lid = get_local_id(0);
          const uint lsize = get_local_size(0);
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
            uint y = Set_HasEdge(vset, edge) ? 1 : 0;
            __global Float* pi_a = Pi(g_pi, u);
            __global Float* pi_b = Pi(g_pi, v);
            Float pi_sum = 0;
            Float probs_sum = 0;

            for (uint k = lid; k < K; k += lsize) {
              Float f = pi_a[k] * pi_b[k];
              scratch[k] = f;
              Float probs_k;
              if (y) {
                probs_k = beta[k] * f;
              } else {
                probs_k = (1.0 - beta[k]) * f;
              }
              probs[k] = probs_k;
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_FOLD_Float(scratch, aux, K);
            pi_sum = scratch[0];
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_Float(probs, scratch, aux, K);
            probs_sum = scratch[0];

            Float prob_0 = (y ? EPSILON : (1.0 - EPSILON)) * (1.0 - pi_sum);
            probs_sum += prob_0;
            for (uint k = lid; k < K; k += lsize) {
              Float f = probs[k] / probs_sum;
              Float one_over_theta_sum = 1.0 / theta_sum[k];
              grads[k] += f * ((1 - y) / theta[k] - one_over_theta_sum);
              grads[k + K] += f * (y / theta[k + K] - one_over_theta_sum);
            }
          }
        }

        __kernel void sum_grads(__global Float* grads, uint num_partial_sums) {
          uint i = get_global_id(0);
          uint gsize = get_global_size(0);
          for (; i < 2 * K; i += gsize) {
            for (uint p = 1; p < num_partial_sums; ++p) {
              grads[i] += grads[i + p * 2 * K];
            }
          }
        }

        __kernel void update_theta(__global Float* theta, __global Float* grads,
                                   uint step_count, Float scale) {
          uint gsize = get_global_size(0);
          Float eps_t = get_eps_t(step_count);
          for (uint k = get_global_id(0); k < K; k += gsize) {
            Float f0 = sqrt(eps_t * theta[k]);
            Float f1 = sqrt(eps_t * theta[k + K]);
            theta[k] = fabs(theta[k] +
                            eps_t / 2.0 * (ETA0 - theta[k] + scale * grads[k]) +
                            f0 * 1 /*randn noise*/);
            theta[k + K] = fabs(
                theta[k + K] +
                eps_t / 2.0 * (ETA1 - theta[k + K] + scale * grads[k + K]) +
                f0 * 1 /*randn noise*/);
          }
        }

        );

BetaUpdater::BetaUpdater(Mode mode, const Config& cfg,
                         compute::command_queue queue,
                         compute::vector<Float>& theta,
                         compute::vector<Float>& beta,
                         compute::vector<Float>& pi, OpenClSet* trainingSet,
                         const std::string& compileFlags,
                         const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      theta_(theta),
      beta_(beta),
      pi_(pi),
      trainingSet_(trainingSet),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.beta_wg_size),
      theta_sum_(cfg.K, queue_.get_context()),
      grads_(cfg.mini_batch_size * 2 * cfg.K, queue_.get_context()),
      probs_(cfg.mini_batch_size * cfg.K, queue_.get_context()) {
  const std::string* src = nullptr;
  switch (mode) {
    case EDGE_PER_THREAD:
      src = &kSourceBeta;
      break;
    case EDGE_PER_WORKGROUP:
      scratch_ = compute::vector<Float>(cfg.mini_batch_size * 2 * cfg.K,
                                        queue_.get_context());
      src = &kSourceBetaWg;
      break;
    default:
      LOG(FATAL) << "Failed to recognize mode";
  }
  prog_ = compute::program::create_with_source(baseFuncs + *src,
                                               queue_.get_context());
  try {
    prog_.build(compileFlags);
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
  }
  theta_sum_kernel_ = prog_.create_kernel("sum_theta");
  theta_sum_kernel_.set_arg(0, theta_);
  theta_sum_kernel_.set_arg(1, theta_sum_);

  grads_partial_kernel_ = prog_.create_kernel("calculate_grads_partial");
  grads_partial_kernel_.set_arg(0, theta_);
  grads_partial_kernel_.set_arg(1, theta_sum_);
  grads_partial_kernel_.set_arg(2, beta_);
  grads_partial_kernel_.set_arg(3, pi_);
  grads_partial_kernel_.set_arg(4, trainingSet->Get());
  grads_partial_kernel_.set_arg(7, probs_);
  grads_partial_kernel_.set_arg(8, grads_);
  if (mode_ == EDGE_PER_WORKGROUP) {
    grads_partial_kernel_.set_arg(9, scratch_);
    grads_partial_kernel_.set_arg(10, local_ * sizeof(Float), 0);
  }

  grads_sum_kernel_ = prog_.create_kernel("sum_grads");
  grads_sum_kernel_.set_arg(0, grads_);

  update_theta_kernel_ = prog_.create_kernel("update_theta");
  update_theta_kernel_.set_arg(0, theta_);
  update_theta_kernel_.set_arg(1, grads_);
}

void BetaUpdater::operator()(compute::vector<Edge>* edges, uint32_t num_edges,
                             Float scale) {
  ++count_calls_;
  {
    theta_sum_event_ =
        queue_.enqueue_1d_range_kernel(theta_sum_kernel_, 0, k_, 0);
    theta_sum_event_.wait();
  }
  {
    uint32_t global = 0;
    grads_partial_kernel_.set_arg(5, *edges);
    grads_partial_kernel_.set_arg(6, num_edges);
    if (mode_ == EDGE_PER_THREAD) {
      global = (num_edges / local_ + (num_edges % local_ ? 1 : 0)) * local_;
      grads_partial_event_ = queue_.enqueue_1d_range_kernel(
          grads_partial_kernel_, 0, global, local_);
    } else {
      global = num_edges * local_;
      grads_partial_event_ = queue_.enqueue_1d_range_kernel(
          grads_partial_kernel_, 0, global, local_);
    }
    grads_partial_event_.wait();

    uint32_t num_partials = std::min(global, num_edges);
    grads_sum_kernel_.set_arg(1, num_partials);
    grads_sum_event_ =
        queue_.enqueue_1d_range_kernel(grads_sum_kernel_, 0, 2 * k_, 0);
    grads_sum_event_.wait();
  }
  {
    update_theta_kernel_.set_arg(2, count_calls_);
    update_theta_kernel_.set_arg(3, scale);
    update_theta_event_ =
        queue_.enqueue_1d_range_kernel(update_theta_kernel_, 0, k_, 0);
    update_theta_event_.wait();
  }
  LOG(INFO)
      << "BetaUpdater: "
      << theta_sum_event_.duration<boost::chrono::nanoseconds>().count() / 1e9
      << ", "
      << grads_partial_event_.duration<boost::chrono::nanoseconds>().count() /
             1e9 << ", "
      << grads_sum_event_.duration<boost::chrono::nanoseconds>().count() / 1e9
      << ", "
      << update_theta_event_.duration<boost::chrono::nanoseconds>().count() /
             1e9;
}

uint64_t BetaUpdater::LastInvocationTime() const {
  return theta_sum_event_.duration<boost::chrono::nanoseconds>().count() +
         grads_partial_event_.duration<boost::chrono::nanoseconds>().count() +
         grads_sum_event_.duration<boost::chrono::nanoseconds>().count() +
         update_theta_event_.duration<boost::chrono::nanoseconds>().count();
}

}  // namespace mcmc
