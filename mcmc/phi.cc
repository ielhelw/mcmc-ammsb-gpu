#include "mcmc/phi.h"

#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourcePhi = BOOST_COMPUTE_STRINGIZE_SOURCE(

    void update_phi_for_node(__global Float* beta, __global Float* g_pi,
                             __global Float* g_phi, __global Set* edge_set,
                             Vertex node, __global Vertex* neighbors,
                             uint step_count,
                             __global Float* grads,  // K
                             __global Float* probs   // K
                             ) {
      __global Float* pi = Pi(g_pi, node);
      __global Float* phi = Pi(g_phi, node);
      Float eps_t = get_eps_t(step_count);
      Float phi_sum = 0;
      for (uint k = 0; k < K; ++k) {
        phi_sum += phi[k];
        // reset grads
        grads[k] = 0;
      }
      for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
        Vertex neighbor = neighbors[i];
        __global Float* pi_neighbor = Pi(g_pi, neighbor);
        Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
        bool y = Set_HasEdge(edge_set, edge);
        Float e = (y == 1 ? EPSILON : 1.0 - EPSILON);
        Float probs_sum = 0;
        for (uint k = 0; k < K; ++k) {
          Float f = (y == 1) ? (beta[k] - EPSILON) : (EPSILON - beta[k]);
          Float probs_k = pi[k] * (pi_neighbor[k] * f + e);
          probs_sum += probs_k;
          probs[k] = probs_k;
        }
        for (uint k = 0; k < K; ++k) {
          grads[k] += (probs[k] / probs_sum) / phi[k] - 1.0 / phi_sum;
        }
      }
      Float Nn = (1.0 * N) / NUM_NEIGHBORS;
      for (uint k = 0; k < K; ++k) {
        Float noise = 1;  // randn();  FIXME
        Float phi_k = phi[k];
        phi[k] = fabs(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                      sqrt(eps_t * phi_k) * noise);
      }
    }

    __kernel void update_phi(
        __global Float* g_beta, __global Float* g_pi, __global Float* g_phi,
        __global void* training_edge_set, __global Vertex* mini_batch_nodes,
        __global Vertex* neighbors,  // [mini_batch_nodes * num_neighbor_sample]
        uint num_mini_batch_nodes, uint step_count,
        __global Float* grads,  // [num global threads * K]
        __global Float* probs   // [num global threads * K]
        ) {
      uint i = get_global_id(0);
      uint gsize = get_global_size(0);
      grads += i * K;
      probs += i * K;
      for (; i < num_mini_batch_nodes; i += gsize) {
        Vertex node = mini_batch_nodes[i];
        __global Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
        update_phi_for_node(g_beta, g_pi, g_phi, training_edge_set, node,
                            node_neighbors, step_count, grads, probs);
      }
    }

    __kernel void update_pi(__global Float* g_pi, __global Float* g_phi,
                            __global Vertex* mini_batch_nodes,
                            uint num_mini_batch_nodes) {
      uint i = get_global_id(0);
      uint gsize = get_global_size(0);
      for (; i < num_mini_batch_nodes; i += gsize) {
        Vertex n = mini_batch_nodes[i];
        __global Float* pi = Pi(g_pi, n);
        __global Float* phi = Pi(g_phi, n);
        Float sum = 0;
        for (uint k = 0; k < K; ++k) {
          sum += phi[k];
        }
        for (uint k = 0; k < K; ++k) {
          pi[k] = phi[k] / sum;
        }
      }
    }

    );

const std::string kSourcePhiWg =
    mcmc::algorithm::WorkGroupNormalizeProgram(compute::type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + compute::type_name<Float>() + "\n"
    "#define WG_NORMALIZE_Float WG_NORMALIZE_" + compute::type_name<Float>() + "\n"
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void update_phi_for_nodeWG(__global Float* beta, __global Float* g_pi,
                                   __global Float* g_phi,
                                   __global Set* edge_set, Vertex node,
                                   __global Vertex* neighbors, uint step_count,
                                   __global Float* grads,    // K
                                   __global Float* probs,    // K
                                   __global Float* scratch,  // K
                                   __local Float* aux) {
          __global Float* pi = Pi(g_pi, node);
          __global Float* phi = Pi(g_phi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = get_local_id(0);
          uint lsize = get_local_size(0);
          // phi sum
          WG_SUM_Float(phi, scratch, aux, K);
          Float phi_sum = scratch[0];
          // reset grads
          for (uint k = lid; k < K; k += lsize) {
            grads[k] = 0;
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            __global Float* pi_neighbor = Pi(g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : 1.0 - EPSILON);
            // probs
            for (uint k = lid; k < K; k += lsize) {
              Float f = (y == 1) ? (beta[k] - EPSILON) : (EPSILON - beta[k]);
              Float probs_k = pi[k] * (pi_neighbor[k] * f + e);
              probs[k] = probs_k;
            }
            // probs sum
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_Float(probs, scratch, aux, K);
            Float probs_sum = scratch[0];
            for (uint k = lid; k < K; k += lsize) {
              grads[k] += (probs[k] / probs_sum) / phi[k] - 1.0 / phi_sum;
            }
          }
          Float Nn = (1.0 * N) / NUM_NEIGHBORS;
          barrier(CLK_GLOBAL_MEM_FENCE);
          for (uint k = lid; k < K; k += lsize) {
            Float noise = 1;  // randn();  FIXME
            Float phi_k = phi[k];
            phi[k] = fabs(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                          sqrt(eps_t * phi_k) * noise);
          }
        }

        __kernel void update_phi(
            __global Float* g_beta, __global Float* g_pi, __global Float* g_phi,
            __global void* training_edge_set, __global Vertex* mini_batch_nodes,
            __global Vertex*
                neighbors,  // [mini_batch_nodes * num_neighbor_sample]
            uint num_mini_batch_nodes,
            uint step_count,
            __global Float* grads,    // [num local threads * K]
            __global Float* probs,    // [num local threads * K]
            __global Float* scratch,  // [num local threads * K]
            __local Float* aux        // [num local threads]
            ) {
          uint i = get_group_id(0);
          uint gsize = get_num_groups(0);
          grads += i * K;
          probs += i * K;
          scratch += i * K;
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex node = mini_batch_nodes[i];
            __global Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
            update_phi_for_nodeWG(g_beta, g_pi, g_phi, training_edge_set, node,
                                  node_neighbors, step_count, grads, probs,
                                  scratch, aux);
          }
        }

        __kernel void update_pi(__global Float* g_pi, __global Float* g_phi,
                                __global Vertex* mini_batch_nodes,
                                uint num_mini_batch_nodes,
                                __global Float* scratch,
                                __local Float* aux) {
          uint i = get_group_id(0);
          uint gsize = get_num_groups(0);
          uint lid = get_local_id(0);
          uint lsize = get_local_size(0);
          scratch += i * K;
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            __global Float* pi = Pi(g_pi, n);
            __global Float* phi = Pi(g_phi, n);
            for (uint k = lid; k < K; k += lsize) {
              pi[k] = phi[k];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_NORMALIZE_Float(pi, scratch, aux, K);
          }
        }

        );

std::ostream& kernel_info(std::ostream& out, compute::kernel& kernel, const compute::device& dev) {
#if 0
  std::vector<size_t> global_size = {0, 0, 0};
  try {
    kernel.get_work_group_info<std::vector<size_t>>(dev, CL_KERNEL_GLOBAL_WORK_SIZE);
  } catch (...) {}
#endif
  out
    << "KERNEL INFO: "
    << kernel.name() << std::endl
#if 0
    << "CL_KERNEL_GLOBAL_WORK_SIZE = "
    << global_size[0] << ", " << global_size[1] << ", " << global_size[2] << std::endl
#endif
    << "CL_KERNEL_WORK_GROUP_SIZE = "
    << kernel.get_work_group_info<size_t>(dev, CL_KERNEL_WORK_GROUP_SIZE) << std::endl
    << "CL_KERNEL_LOCAL_MEM_SIZE = "
    << kernel.get_work_group_info<compute::ulong_>(dev, CL_KERNEL_LOCAL_MEM_SIZE) << std::endl
    << "CL_KERNEL_PRIVATE_MEM_SIZE = "
    << kernel.get_work_group_info<compute::ulong_>(dev, CL_KERNEL_PRIVATE_MEM_SIZE) << std::endl;
  return out;
}

PhiUpdater::PhiUpdater(Mode mode, const Config& cfg, compute::command_queue queue,
                       compute::vector<Float>& beta, compute::vector<Float>& pi,
                       compute::vector<Float>& phi, OpenClSet* trainingSet,
                       const std::string& compileFlags,
                       const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      beta_(beta),
      pi_(pi),
      phi_(phi),
      trainingSet_(trainingSet),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.phi_wg_size),
      grads_(std::min(cfg.N, 2 * cfg.mini_batch_size) * cfg.K, queue_.get_context()),
      probs_(grads_.size(), queue_.get_context()) {
  const std::string* src = nullptr;
  switch (mode_) {
    case NODE_PER_THREAD:
      src = &kSourcePhi;
      break;
    case NODE_PER_WORKGROUP:
      src = &kSourcePhiWg;
      scratch_ = compute::vector<Float>(grads_.size(), queue_.get_context());
      break;
    default:
     LOG(FATAL) << "Cannot recognize mode";
  }
  prog_ = compute::program::create_with_source(baseFuncs + *src,
                                               queue_.get_context());
  try {
    prog_.build(compileFlags);
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
  }
  phi_kernel_ = prog_.create_kernel("update_phi");
  phi_kernel_.set_arg(0, beta);
  phi_kernel_.set_arg(1, pi);
  phi_kernel_.set_arg(2, phi);
  phi_kernel_.set_arg(3, trainingSet->Get());

  phi_kernel_.set_arg(8, grads_);
  phi_kernel_.set_arg(9, probs_);
  if (mode_ == NODE_PER_WORKGROUP) {
    phi_kernel_.set_arg(10, scratch_);
    phi_kernel_.set_arg(11, cfg.K * sizeof(Float), 0);
  }

  pi_kernel_ = prog_.create_kernel("update_pi");
  pi_kernel_.set_arg(0, pi_);
  pi_kernel_.set_arg(1, phi_);
  if (mode_ == NODE_PER_WORKGROUP) {
    pi_kernel_.set_arg(4, scratch_);
    pi_kernel_.set_arg(5, cfg.K * sizeof(Float), 0);
  }
  kernel_info(LOG(INFO), phi_kernel_, queue_.get_device());
  kernel_info(LOG(INFO), pi_kernel_, queue_.get_device());
}

void PhiUpdater::operator()(
    compute::vector<Vertex>& mini_batch_nodes,  // [X <= 2*MINI_BATCH_SIZE]
    compute::vector<Vertex>& neighbors,  // [MINI_BATCH_NODES, NUM_NEIGHBORS]
    uint32_t num_mini_batch_nodes) {
  LOG_IF(FATAL, grads_.size() < num_mini_batch_nodes * k_) << "grads too small";
  LOG_IF(FATAL, probs_.size() < num_mini_batch_nodes * k_) << "probs too small";
  LOG_IF(FATAL, mode_ == NODE_PER_WORKGROUP && scratch_.size() < num_mini_batch_nodes * k_) << "scratch too small";
  ++count_calls_;
  uint32_t global = (num_mini_batch_nodes / local_ +
    (num_mini_batch_nodes % local_ ? 1 : 0)) * local_;
  phi_kernel_.set_arg(4, mini_batch_nodes);
  phi_kernel_.set_arg(5, neighbors);
  phi_kernel_.set_arg(6, num_mini_batch_nodes);
  phi_kernel_.set_arg(7, count_calls_);
  event_ = queue_.enqueue_1d_range_kernel(phi_kernel_, 0, global, local_);
  event_.wait();
  LOG(INFO) << event_.duration<boost::chrono::nanoseconds>().count();
  pi_kernel_.set_arg(2, mini_batch_nodes);
  pi_kernel_.set_arg(3, num_mini_batch_nodes);
  event_ = queue_.enqueue_1d_range_kernel(pi_kernel_, 0, global, local_);
  event_.wait();
  LOG(INFO) << event_.duration<boost::chrono::nanoseconds>().count();
}

}  // namespace mcmc
