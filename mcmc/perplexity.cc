#include "mcmc/perplexity.h"

#include <boost/compute/algorithm/reduce.hpp>
#include <glog/logging.h>
#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourcePerplexity = BOOST_COMPUTE_STRINGIZE_SOURCE(

    Float calculate_edge_likelihood(__global Float * pi_a, __global Float* pi_b,
                                    __global Float* beta, bool is_edge) {
      Float s = 0;
      if (is_edge) {
        uint k = 0;
        for (; k < K; ++k) {
          s += pi_a[k] * pi_b[k] * Beta(beta, k);
        }
      } else {
        Float sum = 0;
        uint k = 0;
        for (; k < K; ++k) {
          Float f = pi_a[k] * pi_b[k];
          s += f * (1.0 - Beta(beta, k));
          sum += f;
        }
        s += (1.0 - sum) * (1.0 - EPSILON);
      }
      if (s < 1.0e-30) {
        s = 1.0e-30;
      }
      return s;
    }

    void calculate_ppx_partial_for_edge_(
        __global void* g_pi, __global Float* g_beta, __global Set* edge_set,
        __global Float* g_ppx_per_edge, __global Float* g_link_likelihood,
        __global Float* g_non_link_likelihood, __global uint* g_link_count,
        __global uint* g_non_link_count, uint avg_count, Edge e) {
      Vertex u = Vertex0(e);
      Vertex v = Vertex1(e);
      bool is_edge = Set_HasEdge(edge_set, e);
      Float edge_likelihood = calculate_edge_likelihood(
          FloatRowPartitionedMatrix_Row(g_pi, u),
          FloatRowPartitionedMatrix_Row(g_pi, v), g_beta, is_edge);
      Float ppx = *g_ppx_per_edge;
      ppx = (ppx * (avg_count - 1) + edge_likelihood) / avg_count;
      if (is_edge) {
        *g_link_count = 1;
        *g_link_likelihood = log(ppx);
        *g_non_link_count = 0;
        *g_non_link_likelihood = 0;
      } else {
        (*g_non_link_count) = 1;
        *g_non_link_likelihood = log(ppx);
        *g_link_count = 0;
        *g_link_likelihood = 0;
      }
      *g_ppx_per_edge = ppx;
    }

    __kernel void calculate_ppx_partial_for_edge(
        __global Edge* edges, uint num_edges, __global void* g_pi,
        __global Float* g_beta, __global void* /* Set* */ void_edge_set,
        __global Float* g_ppx_per_edge,         // [num global threads]
        __global Float* g_link_likelihood,      // [num global threads]
        __global Float* g_non_link_likelihood,  // [num global threads]
        __global uint* g_link_count,            // [num global threads]
        __global uint* g_non_link_count,        // [num global threads]
        uint avg_count) {
      size_t i = get_global_id(0);
      for (; i < num_edges; i += get_global_size(0)) {
        calculate_ppx_partial_for_edge_(
            g_pi, g_beta, void_edge_set, g_ppx_per_edge + i,
            g_link_likelihood + i, g_non_link_likelihood + i, g_link_count + i,
            g_non_link_count + i, avg_count, edges[i]);
      }
    }

    );

const std::string kSourcePerplexityWg =
    mcmc::algorithm::WorkGroupSum(compute::type_name<Float>()) + "\n" +
    "#define WG_SUM_FOLD_Float WG_SUM_FOLD_" + compute::type_name<Float>() +
    "\n" BOOST_COMPUTE_STRINGIZE_SOURCE(

        Float calculate_edge_likelihood_WG(
            __global Float * pi_a, __global Float* pi_b, __global Float* beta,
            bool is_edge, __global Float* scratch, __local Float* aux) {
          uint lid = get_local_id(0);
          Float s = 0;
          if (is_edge) {
            for (uint i = lid; i < K; i += get_local_size(0)) {
              scratch[i] = pi_a[i] * pi_b[i] * Beta(beta, i);
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_FOLD_Float(scratch, aux, K);
            s = scratch[0];
          } else {
            Float sum = 0;
            for (uint i = lid; i < K; i += get_local_size(0)) {
              scratch[i] = pi_a[i] * pi_b[i];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_FOLD_Float(scratch, aux, K);
            sum = scratch[0];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint i = lid; i < K; i += get_local_size(0)) {
              scratch[i] = pi_a[i] * pi_b[i] * (1.0 - Beta(beta, i));
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_FOLD_Float(scratch, aux, K);
            s = scratch[0];
            s += (1.0 - sum) * (1.0 - EPSILON);
          }
          if (s < 1.0e-30) {
            s = 1.0e-30;
          }
          return s;
        }

        void calculate_ppx_partial_for_edge_(
            __global void* g_pi, __global Float* g_beta, __global Set* edge_set,
            __global Float* g_ppx_per_edge, __global Float* g_link_likelihood,
            __global Float* g_non_link_likelihood, __global uint* g_link_count,
            __global uint* g_non_link_count, uint avg_count, Edge e,
            __global Float* scratch, __local Float* aux) {
          Vertex u = Vertex0(e);
          Vertex v = Vertex1(e);
          bool is_edge = Set_HasEdge(edge_set, e);
          Float edge_likelihood = calculate_edge_likelihood_WG(
              FloatRowPartitionedMatrix_Row(g_pi, u),
              FloatRowPartitionedMatrix_Row(g_pi, v), g_beta, is_edge, scratch,
              aux);
          if (get_local_id(0) == 0) {
            Float ppx = *g_ppx_per_edge;
            ppx = (ppx * (avg_count - 1) + edge_likelihood) / avg_count;
            if (is_edge) {
              *g_link_count = 1;
              *g_link_likelihood = log(ppx);
              *g_non_link_count = 0;
              *g_non_link_likelihood = 0;
            } else {
              *g_non_link_count = 1;
              *g_non_link_likelihood = log(ppx);
              *g_link_count = 0;
              *g_link_likelihood = 0;
            }
            *g_ppx_per_edge = ppx;
          }
        }

        __kernel void calculate_ppx_partial_for_edge(
            __global Edge* edges, uint num_edges, __global void* g_pi,
            __global Float* g_beta, __global void* /* Set* */ void_edge_set,
            __global Float* g_ppx_per_edge,         // [num work groups]
            __global Float* g_link_likelihood,      // [num work groups]
            __global Float* g_non_link_likelihood,  // [num work groups]
            __global uint* g_link_count,            // [num work groups]
            __global uint* g_non_link_count,        // [num work groups]
            uint avg_count,
            __global Float* scratch,  // [num work groups * K]
            __local Float* aux) {
          size_t i = get_group_id(0);
          scratch += get_group_id(0) * K;
          for (; i < num_edges; i += get_num_groups(0)) {
            calculate_ppx_partial_for_edge_(
                g_pi, g_beta, void_edge_set, g_ppx_per_edge + i,
                g_link_likelihood + i, g_non_link_likelihood + i,
                g_link_count + i, g_non_link_count + i, avg_count, edges[i],
                scratch, aux);
          }
        }

        );

PerplexityCalculator::PerplexityCalculator(
    Mode mode, const Config& cfg, compute::command_queue queue,
    compute::vector<Float>& beta, RowPartitionedMatrix<Float>* pi,
    compute::vector<Edge>& edges, OpenClSet* edgeSet,
    const std::string& compileFlags, const std::string& baseFuncs)
    : queue_(queue),
      beta_(beta),
      pi_(pi),
      edges_(edges),
      edgeSet_(edgeSet),
      ppx_per_edge_(edges_.size(), 0, queue_),
      ppx_per_edge_link_likelihood_(edges.size(), 0, queue_),
      ppx_per_edge_non_link_likelihood_(edges.size(), 0, queue_),
      ppx_per_edge_link_count_(edges.size(), static_cast<compute::uint_>(0),
                               queue_),
      ppx_per_edge_non_link_count_(edges.size(), static_cast<compute::uint_>(0),
                                   queue_),
      link_count_(1),
      non_link_count_(1),
      link_likelihood_(1),
      non_link_likelihood_(1),
      count_calls_(0),
      local_(cfg.ppx_wg_size) {
  const std::string* src = nullptr;
  switch (mode) {
    case EDGE_PER_THREAD:
      src = &kSourcePerplexity;
      global_ =
          (edges_.size() / local_ + (edges_.size() % local_ ? 1 : 0)) * local_;
      break;
    case EDGE_PER_WORKGROUP:
      src = &kSourcePerplexityWg;
      global_ = edges.size() * local_;
      break;
    default:
      LOG(FATAL) << "Cannot recognize mode: " << mode;
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
  // ppx_kernel
  kernel_ = prog_.create_kernel("calculate_ppx_partial_for_edge");
  kernel_.set_arg(0, edges_);
  kernel_.set_arg(1, static_cast<compute::uint_>(edges_.size()));
  kernel_.set_arg(2, pi_->Get());
  kernel_.set_arg(3, beta_);
  kernel_.set_arg(4, edgeSet_->Get());
  kernel_.set_arg(5, ppx_per_edge_);
  kernel_.set_arg(6, ppx_per_edge_link_likelihood_);
  kernel_.set_arg(7, ppx_per_edge_non_link_likelihood_);
  kernel_.set_arg(8, ppx_per_edge_link_count_);
  kernel_.set_arg(9, ppx_per_edge_non_link_count_);
  // kernel_.set_arg(10, count_calls_);
  if (mode == EDGE_PER_WORKGROUP) {
    scratch_ =
        compute::vector<Float>(edges_.size() * cfg.K, queue_.get_context()),
    kernel_.set_arg(11, scratch_);
    kernel_.set_arg(12, local_ * sizeof(Float), 0);
  }
}

uint64_t PerplexityCalculator::LastInvocationTime() const {
  return event_.duration<boost::chrono::nanoseconds>().count();
}

Float PerplexityCalculator::operator()() {
  count_calls_++;
  kernel_.set_arg(10, count_calls_);
  event_ = queue_.enqueue_1d_range_kernel(kernel_, 0, global_, local_);
  event_.wait();
  compute::reduce(ppx_per_edge_link_count_.begin(),
                  ppx_per_edge_link_count_.end(), link_count_.begin(), queue_);
  compute::reduce(ppx_per_edge_non_link_count_.begin(),
                  ppx_per_edge_non_link_count_.end(), non_link_count_.begin(),
                  queue_);
  compute::reduce(ppx_per_edge_link_likelihood_.begin(),
                  ppx_per_edge_link_likelihood_.end(), link_likelihood_.begin(),
                  queue_);
  compute::reduce(ppx_per_edge_non_link_likelihood_.begin(),
                  ppx_per_edge_non_link_likelihood_.end(),
                  non_link_likelihood_.begin(), queue_);
  double avg_likelihood = 0.0;
  if (link_count_[0] + non_link_count_[0] != 0) {
    avg_likelihood = (link_likelihood_[0] + non_link_likelihood_[0]) /
                     (link_count_[0] + non_link_count_[0]);
  }
  LOG(INFO) << "link_count " << link_count_[0];
  LOG(INFO) << "non_link_count " << non_link_count_[0];
  LOG(INFO) << "link_likelihood " << link_likelihood_[0];
  LOG(INFO) << "non_link_likelihood " << non_link_likelihood_[0];
  return (-avg_likelihood);
}

}  // namespace mcmc
