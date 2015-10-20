#include "mcmc/phi.h"

#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourcePhi =
    random::GetRandomHeader() +
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void update_phi_for_node(GLOBAL Float* beta, GLOBAL void* g_pi,
                                 GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                 GLOBAL Set* edge_set, Vertex node,
                                 GLOBAL Vertex* neighbors, uint step_count,
                                 GLOBAL Float* grads,  // K
                                 GLOBAL Float* probs,  // K
                                 random_seed_t* rseed) {
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row(g_pi, node);
          Float eps_t = get_eps_t(step_count);
          Float phi_sum = g_phi[node];
          for (uint k = 0; k < K; ++k) {
            // reset grads
            grads[k] = 0;
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row(g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : 1.0 - EPSILON);
            Float probs_sum = 0;
            for (uint k = 0; k < K; ++k) {
              Float f = (y == 1) ? (Beta(beta, k) - EPSILON)
                                 : (EPSILON - Beta(beta, k));
              Float probs_k = pi[k] * (pi_neighbor[k] * f + e);
              probs_sum += probs_k;
              probs[k] = probs_k;
            }
            for (uint k = 0; k < K; ++k) {
              grads[k] +=
                  (probs[k] / probs_sum) / (pi[k] * phi_sum) - 1.0 / phi_sum;
            }
          }
          Float Nn = (1.0 * N) / NUM_NEIGHBORS;
          for (uint k = 0; k < K; ++k) {
            Float noise = PHI_RANDN(rseed);
            Float phi_k = pi[k] * phi_sum;
            phi_vec[k] =
                fabs(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                     sqrt(eps_t * phi_k) * noise);
          }
        }

        KERNEL void update_phi(
            GLOBAL Float* g_beta, GLOBAL void* g_pi, GLOBAL Float* g_phi,
            GLOBAL Float* g_phi_vec, GLOBAL void* training_edge_set,
            GLOBAL Vertex* mini_batch_nodes,
            GLOBAL Vertex*
                neighbors,  // [mini_batch_nodes * num_neighbor_sample]
            uint num_mini_batch_nodes,
            uint step_count,
            GLOBAL Float* grads,  // [num global threads * K]
            GLOBAL Float* probs,  // [num global threads * K]
            GLOBAL void* vrand) {
          uint i = GET_GLOBAL_ID();
          uint gsize = GET_GLOBAL_SIZE();
          grads += i * K;
          probs += i * K;
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_node(g_beta, g_pi, g_phi, phi_vec,
                                  training_edge_set, node, node_neighbors,
                                  step_count, grads, probs, &rseed);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes) {
          uint i = GET_GLOBAL_ID();
          uint gsize = GET_GLOBAL_SIZE();
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row(g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
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
    mcmc::algorithm::WorkGroupNormalizeProgram(compute::type_name<Float>()) +
    "\n" + "#define WG_SUM_Float WG_SUM_" + compute::type_name<Float>() +
    "\n"
    "#define WG_NORMALIZE_Float WG_NORMALIZE_" +
    compute::type_name<Float>() + "\n" + random::GetRandomHeader() +
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   GLOBAL Float* grads,  // K
                                   GLOBAL Float* probs,  // K
                                   random_seed_t* rseed,
                                   GLOBAL Float* scratch,  // K
                                   LOCAL Float* aux) {
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row(g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          uint lsize = GET_LOCAL_SIZE();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          for (uint k = lid; k < K; k += lsize) {
            grads[k] = 0;
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row(g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : 1.0 - EPSILON);
            // probs
            for (uint k = lid; k < K; k += lsize) {
              Float f = (y == 1) ? (Beta(beta, k) - EPSILON)
                                 : (EPSILON - Beta(beta, k));
              Float probs_k = pi[k] * (pi_neighbor[k] * f + e);
              probs[k] = probs_k;
            }
            // probs sum
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_SUM_Float(probs, aux, K);
            Float probs_sum = aux[0];
            for (uint k = lid; k < K; k += lsize) {
              grads[k] +=
                  (probs[k] / probs_sum) / (pi[k] * phi_sum) - 1.0 / phi_sum;
            }
          }
          Float Nn = (1.0 * N) / NUM_NEIGHBORS;
          barrier(CLK_GLOBAL_MEM_FENCE);
          for (uint k = lid; k < K; k += lsize) {
            Float noise = PHI_RANDN(rseed);
            Float phi_k = pi[k] * phi_sum;
            phi_vec[k] =
                fabs(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                     sqrt(eps_t * phi_k) * noise);
          }
        }

        KERNEL void update_phi(
            GLOBAL Float* g_beta, GLOBAL void* g_pi, GLOBAL Float* g_phi,
            GLOBAL Float* g_phi_vec, GLOBAL void* training_edge_set,
            GLOBAL Vertex* mini_batch_nodes,
            GLOBAL Vertex*
                neighbors,  // [mini_batch_nodes * num_neighbor_sample]
            uint num_mini_batch_nodes,
            uint step_count,
            GLOBAL Float* grads,  // [num local threads * K]
            GLOBAL Float* probs,  // [num local threads * K]
            GLOBAL void* vrand,
            GLOBAL Float* scratch,  // [num local threads * K]
            LOCAL Float* aux        // [num local threads]
            ) {
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          grads += i * K;
          probs += i * K;
          scratch += i * K;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, phi_vec,
                                    training_edge_set, node, node_neighbors,
                                    step_count, grads, probs, &rseed, scratch,
                                    aux);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes, GLOBAL Float* scratch,
                              LOCAL Float* aux) {
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          uint lid = GET_LOCAL_ID();
          uint lsize = GET_LOCAL_SIZE();
          scratch += i * K;
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row(g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
            for (uint k = lid; k < K; k += lsize) {
              pi[k] = phi[k];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
            WG_NORMALIZE_Float(pi, aux, K);
          }
        }

        );

PhiUpdater::PhiUpdater(Mode mode, const Config& cfg,
                       compute::command_queue queue,
                       compute::vector<Float>& beta,
                       RowPartitionedMatrix<Float>* pi,
                       compute::vector<Float>& phi, OpenClSet* trainingSet,
                       const std::string& compileFlags,
                       const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      beta_(beta),
      pi_(pi),
      phi_(phi),
      phi_vec(2 * cfg.mini_batch_size * cfg.K, queue_.get_context()),
      trainingSet_(trainingSet),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(
          2 * cfg.mini_batch_size *
              (mode == NODE_PER_THREAD ? 1 : cfg.phi_wg_size),
          random::random_seed_t{42, 43})),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.phi_wg_size),
      grads_(2 * cfg.mini_batch_size * cfg.K, queue_.get_context()),
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
  std::ostringstream out;
  if (cfg.phi_disable_noise) {
    out << "#define PHI_RANDN(X) 1" << std::endl;
  } else {
    out << "#define PHI_RANDN(X) randn(X)" << std::endl;
  }
  out << baseFuncs << std::endl << GetRowPartitionedMatrixHeader<Float>()
      << std::endl << "#define FloatRowPartitionedMatrix_Row "
      << compute::type_name<Float>() << "RowPartitionedMatrix_Row\n" << *src
      << std::endl;
  prog_ = compute::program::create_with_source(out.str(), queue_.get_context());
  try {
    prog_.build(compileFlags);
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
  }
  LOG(INFO) << "####################### PHI LOG:" << std::endl
            << prog_.build_log();
  phi_kernel_ = prog_.create_kernel("update_phi");
  phi_kernel_.set_arg(0, beta_);
  phi_kernel_.set_arg(1, pi_->Get());
  phi_kernel_.set_arg(2, phi_);
  phi_kernel_.set_arg(3, phi_vec);
  phi_kernel_.set_arg(4, trainingSet->Get()());

  phi_kernel_.set_arg(9, grads_);
  phi_kernel_.set_arg(10, probs_);
  phi_kernel_.set_arg(11, rand_->Get());
  if (mode_ == NODE_PER_WORKGROUP) {
    phi_kernel_.set_arg(12, scratch_);
    phi_kernel_.set_arg(13, local_ * sizeof(Float), 0);
  }

  pi_kernel_ = prog_.create_kernel("update_pi");
  pi_kernel_.set_arg(0, pi_->Get());
  pi_kernel_.set_arg(1, phi_vec);
  if (mode_ == NODE_PER_WORKGROUP) {
    pi_kernel_.set_arg(4, scratch_);
    pi_kernel_.set_arg(5, local_ * sizeof(Float), 0);
  }
}

void PhiUpdater::operator()(
    compute::vector<Vertex>& mini_batch_nodes,  // [X <= 2*MINI_BATCH_SIZE]
    compute::vector<Vertex>& neighbors,  // [MINI_BATCH_NODES, NUM_NEIGHBORS]
    uint32_t num_mini_batch_nodes) {
  LOG_IF(FATAL, num_mini_batch_nodes == 0) << "mini-batch nodes size = 0!";
  LOG_IF(FATAL, grads_.size() < num_mini_batch_nodes * k_) << "grads too small";
  LOG_IF(FATAL, probs_.size() < num_mini_batch_nodes * k_) << "probs too small";
  LOG_IF(FATAL, mode_ == NODE_PER_WORKGROUP &&
                    scratch_.size() < num_mini_batch_nodes * k_)
      << "scratch too small";
  ++count_calls_;
  uint32_t global;
  if (mode_ == NODE_PER_THREAD) {
    global = (num_mini_batch_nodes / local_ +
              (num_mini_batch_nodes % local_ ? 1 : 0)) *
             local_;
  } else {
    global = num_mini_batch_nodes * local_;
  }
  LOG_IF(FATAL, rand_->GetSeeds().size() < global)
      << "Num seeds smaller than global threads";
  phi_kernel_.set_arg(5, mini_batch_nodes);
  phi_kernel_.set_arg(6, neighbors);
  phi_kernel_.set_arg(7, num_mini_batch_nodes);
  phi_kernel_.set_arg(8, count_calls_);
  phi_event_ = queue_.enqueue_1d_range_kernel(phi_kernel_, 0, global, local_);
  phi_event_.wait();
  //  LOG(INFO) << phi_event_.duration<boost::chrono::nanoseconds>().count();
  pi_kernel_.set_arg(2, mini_batch_nodes);
  pi_kernel_.set_arg(3, num_mini_batch_nodes);
  pi_event_ = queue_.enqueue_1d_range_kernel(pi_kernel_, 0, global, local_);
  pi_event_.wait();
  //  LOG(INFO) << pi_event_.duration<boost::chrono::nanoseconds>().count();
}

uint64_t PhiUpdater::LastInvocationTime() const {
  return phi_event_.duration<boost::chrono::nanoseconds>().count() +
         pi_event_.duration<boost::chrono::nanoseconds>().count();
}

}  // namespace mcmc
