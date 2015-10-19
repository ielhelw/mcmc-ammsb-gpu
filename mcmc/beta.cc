#include "mcmc/beta.h"

#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string kSourceBetaBase =
    random::GetRandomHeader() +
    BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void sum_theta(__global Float* g_theta,     // [K, 2]
                                __global Float* g_theta_sum  // [K]
                                ) {
          uint gsize = get_global_size(0);
          for (uint i = get_global_id(0); i < K; i += gsize) {
            g_theta_sum[i] = Theta0(g_theta, i) + Theta1(g_theta,  i);
          }
        } __kernel void sum_grads(__global Float* grads,
                                  uint num_partial_sums) {
          uint i = get_global_id(0);
          uint gsize = get_global_size(0);
          for (; i < 2 * K; i += gsize) {
            for (uint p = 1; p < num_partial_sums; ++p) {
              grads[i] += grads[i + p * 2 * K];
            }
          }
        }

        __kernel void update_theta(__global Float* theta, __global Float* grads,
                                   uint step_count, Float scale,
                                   __global void* vrand) {
          const uint gid = get_global_id(0);
          const uint gsize = get_global_size(0);
          __global Random* random = (__global Random*)vrand;
          if (gid < K) {
            random_seed_t rseed = random->base_[gid];
            Float eps_t = get_eps_t(step_count);
            for (uint k = gid; k < K; k += gsize) {
              Float r0 = randn(&rseed);
              Float grads_k = grads[2*k];
              Float theta_k = Theta0(theta, k);
              Float f0 = sqrt(eps_t * theta_k);
              SetTheta0(theta, k, fabs(theta_k +
                              eps_t / 2.0 * (ETA0 - theta_k + scale * grads_k) +
                              f0 * r0));
              Float r1 = randn(&rseed);
              Float grads_2k = grads[2*k + 1];
              Float theta_2k = Theta1(theta, k);
              Float f1 = sqrt(eps_t * theta_2k);
              SetTheta1(theta, k, fabs(
                  theta_2k +
                  eps_t / 2.0 * (ETA1 - theta_2k + scale * grads_2k) + f1 * r1));
            }
            random->base_[gid] = rseed;
          }
        }

        );

const std::string kSourceBeta =
    kSourceBetaBase +
    BOOST_COMPUTE_STRINGIZE_SOURCE(__kernel void calculate_grads_partial(
        __global Float* theta,      // [K, 2]
        __global Float* theta_sum,  // [K]
        __global Float* beta,       // [K]
        __global void* g_pi,        // [N, K]
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
        __global Float* pi_a = FloatRowPartitionedMatrix_Row(g_pi, u);
        __global Float* pi_b = FloatRowPartitionedMatrix_Row(g_pi, v);
        Float pi_sum = 0;
        Float probs_sum = 0;

        for (uint k = 0; k < K; ++k) {
          Float f = pi_a[k] * pi_b[k];
          pi_sum += f;
          Float probs_k;
          if (y) {
            probs_k = Beta(beta, k) * f;
          } else {
            probs_k = (1.0 - Beta(beta, k)) * f;
          }
          probs[k] = probs_k;
          probs_sum += probs_k;
        }
        Float prob_0 = (y ? EPSILON : (1.0 - EPSILON)) * (1.0 - pi_sum);
        probs_sum += prob_0;
        for (uint k = 0; k < K; k++) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = 1.0 / theta_sum[k];
          grads[2*k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);
          grads[2*k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);
        }
      }
    });

const std::string kSourceBetaWg =
    mcmc::algorithm::WorkGroupSum(compute::type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" +
    compute::type_name<Float>() + "\n" + kSourceBetaBase +
    BOOST_COMPUTE_STRINGIZE_SOURCE(__kernel void calculate_grads_partial(
        __global Float* theta,      // [K, 2]
        __global Float* theta_sum,  // [K]
        __global Float* beta,       // [K]
        __global void* g_pi,        // [N, K]
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
        __global Float* pi_a = FloatRowPartitionedMatrix_Row(g_pi, u);
        __global Float* pi_b = FloatRowPartitionedMatrix_Row(g_pi, v);
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
            probs_k = (1.0 - beta_k) * f;
          }
          probs[k] = probs_k;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        WG_SUM_Float(scratch, aux, K);
        pi_sum = aux[0];
        barrier(CLK_GLOBAL_MEM_FENCE);
        WG_SUM_Float(probs, aux, K);
        probs_sum = aux[0];

        Float prob_0 = (y ? EPSILON : (1.0 - EPSILON)) * (1.0 - pi_sum);
        probs_sum += prob_0;
        for (uint k = lid; k < K; k += lsize) {
          Float f = probs[k] / probs_sum;
          Float one_over_theta_sum = 1.0 / theta_sum[k];
          grads[2*k] += f * ((1 - y) / Theta0(theta, k) - one_over_theta_sum);
          grads[2*k + 1] += f * (y / Theta1(theta, k) - one_over_theta_sum);
        }
      }
    });

BetaUpdater::BetaUpdater(
    Mode mode, const Config& cfg, compute::command_queue queue,
    compute::vector<Float>& theta, compute::vector<Float>& beta,
    RowPartitionedMatrix<Float>* pi, OpenClSet* trainingSet,
    const std::string& compileFlags, const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      theta_(theta),
      beta_(beta),
      pi_(pi),
      trainingSet_(trainingSet),
      normalizer_(queue_, &beta_, 2, 1),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(cfg.K, random::random_seed_t{42, 43})),
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
  prog_ = compute::program::create_with_source(
      baseFuncs + GetRowPartitionedMatrixHeader<Float>() +
          "#define FloatRowPartitionedMatrix_Row " +
          compute::type_name<Float>() + "RowPartitionedMatrix_Row\n" + *src,
      queue_.get_context());
  try {
    prog_.build(compileFlags);
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
  }
  LOG(INFO) << "####################### BETA LOG:" << std::endl << prog_.build_log();
  theta_sum_kernel_ = prog_.create_kernel("sum_theta");
  theta_sum_kernel_.set_arg(0, theta_);
  theta_sum_kernel_.set_arg(1, theta_sum_);

  grads_partial_kernel_ = prog_.create_kernel("calculate_grads_partial");
  grads_partial_kernel_.set_arg(0, theta_);
  grads_partial_kernel_.set_arg(1, theta_sum_);
  grads_partial_kernel_.set_arg(2, beta_);
  grads_partial_kernel_.set_arg(3, pi_->Get());
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
  update_theta_kernel_.set_arg(4, rand_->Get());
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
  {
    auto t1 = std::chrono::high_resolution_clock::now();
    compute::copy(theta_.begin(), theta_.end(), beta_.begin(), queue_);
    normalizer_();
    auto t2 = std::chrono::high_resolution_clock::now();
    normalize_time_ =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t2).count();
  }
#if 0
  LOG(INFO)
      << "BetaUpdater: "
      << theta_sum_event_.duration<boost::chrono::nanoseconds>().count() / 1e9
      << ", "
      << grads_partial_event_.duration<boost::chrono::nanoseconds>().count() /
             1e9 << ", "
      << grads_sum_event_.duration<boost::chrono::nanoseconds>().count() / 1e9
      << ", "
      << update_theta_event_.duration<boost::chrono::nanoseconds>().count() /
             1e9 << ", " << normalize_time_ / 1e9;
#endif
}

uint64_t BetaUpdater::LastInvocationTime() const {
  return theta_sum_event_.duration<boost::chrono::nanoseconds>().count() +
         grads_partial_event_.duration<boost::chrono::nanoseconds>().count() +
         grads_sum_event_.duration<boost::chrono::nanoseconds>().count() +
         update_theta_event_.duration<boost::chrono::nanoseconds>().count();
}

}  // namespace mcmc
