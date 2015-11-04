#include "mcmc/phi.h"

#include <boost/algorithm/string/replace.hpp>
#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourcePhi = random::GetRandomHeader() + R"%%(

        void update_phi_for_node(GLOBAL Float* beta, GLOBAL void* g_pi,
                                 GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                 GLOBAL Set* edge_set, Vertex node,
                                 GLOBAL Vertex* neighbors, uint step_count,
                                 GLOBAL Float* grads,  // K
                                 GLOBAL Float* probs,  // K
                                 random_seed_t* rseed) {
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          Float phi_sum = g_phi[node];
          for (uint k = 0; k < K; ++k) {
            // reset grads
            grads[k] = 0;
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
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
                  (probs[k] / probs_sum) / (pi[k] * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint k = 0; k < K; ++k) {
            Float noise = PHI_RANDN(rseed);
            Float phi_k = pi[k] * phi_sum;
            phi_vec[k] =
                FABS(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                     SQRT(eps_t * phi_k) * noise);
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
                                  (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                  step_count, grads, probs, &rseed);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Float* g_phi,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes) {
          uint i = GET_GLOBAL_ID();
          uint gsize = GET_GLOBAL_SIZE();
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
            Float sum = 0;
            for (uint k = 0; k < K; ++k) {
              sum += phi[k];
            }
            for (uint k = 0; k < K; ++k) {
              pi[k] = phi[k] / sum;
            }
            g_phi[n] = sum;
          }
        }

        )%%";
#if 0  // pi_a/probs/grads in thread local memory. Not enough registers, spills.
 const std::string kSourcePhiWg =
    mcmc::algorithm::WorkGroupNormalizeProgram(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() +
    "\n"
    "#define WG_SUM_Float_LOCAL_ WG_SUM_" +
    type_name<Float>() +
    "_LOCAL_\n"
    "#define WG_SUML_Float WG_SUML_" +
    type_name<Float>() +
    "\n"
    "#define WG_NORMALIZE_Float WG_NORMALIZE_" +
    type_name<Float>() + "\n" + random::GetRandomHeader() + R"%%(

        #define K_PER_THREAD ((K/WG_SIZE) + (K % WG_SIZE? 1 : 0))

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux) {
          Float grads[K_PER_THREAD];
          Float probs[K_PER_THREAD];
          Float pi_a[K_PER_THREAD];
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
            grads[i] = 0;
            pi_a[i] = pi[k];
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            // probs
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
                Float beta_k = Beta(beta, k);
                Float f = (y == 1) ? (beta_k - EPSILON)
                                   : (EPSILON - beta_k);
                Float pi_k = pi_a[i];
                Float pin_k = pi_neighbor[k];
                probs[i] = pi_k * (pin_k * f + e);
            }
            // probs sum
            Float probs_sum = 0;
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
              probs_sum += probs[i];
            }
            aux[lid] = probs_sum;
            BARRIER_LOCAL;
            WG_SUM_Float_LOCAL_(aux, K);
            probs_sum = aux[0];
            BARRIER_LOCAL;
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
                Float pi_k = pi_a[i];
                Float probs_k = probs[i];
                grads[i] += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
              Float noise = PHI_RANDN(rseed);
              Float phi_k = pi_a[i] * phi_sum;
              phi_vec[k] =
                  FABS(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[i]) +
                       SQRT(eps_t * phi_k) * noise);
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
            GLOBAL void* vrand) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Float* g_phi,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          uint lid = GET_LOCAL_ID();
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
            for (uint k = lid; k < K; k += WG_SIZE) {
              pi[k] = phi[k];
            }
            BARRIER_GLOBAL;
            WG_NORMALIZE_Float(pi, aux, K);
            if (lid == 0) g_phi[n] = aux[0];
          }
        }

        )%%";
#elif 0
// 25% improvement by placing pi_a/probs/grads in shared memory.
const std::string kSourcePhiWg =
    mcmc::algorithm::WorkGroupNormalizeProgram(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() +
    "\n"
    "#define WG_SUM_Float_LOCAL_ WG_SUM_" +
    type_name<Float>() +
    "_LOCAL_\n"
    "#define WG_SUML_Float WG_SUML_" +
    type_name<Float>() +
    "\n"
    "#define WG_NORMALIZE_Float WG_NORMALIZE_" +
    type_name<Float>() + "\n" + random::GetRandomHeader() + R"%%(

        #define K_PER_THREAD ((K/WG_SIZE) + (K % WG_SIZE? 1 : 0))

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux,
                                   LOCAL Float* grads,
                                   LOCAL Float* probs,
                                   LOCAL Float* pi_a
                                   ) {
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
            grads[k] = 0;
            pi_a[k] = pi[k];
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            // probs
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
                Float beta_k = Beta(beta, k);
                Float f = (y == 1) ? (beta_k - EPSILON)
                                   : (EPSILON - beta_k);
                Float pi_k = pi_a[k];
                Float pin_k = pi_neighbor[k];
                probs[k] = pi_k * (pin_k * f + e);
            }
            // probs sum
            Float probs_sum = 0;
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
              probs_sum += probs[k];
            }
            aux[lid] = probs_sum;
            BARRIER_LOCAL;
            WG_SUM_Float_LOCAL_(aux, K);
            probs_sum = aux[0];
            BARRIER_LOCAL;
            for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
                Float pi_k = pi_a[k];
                Float probs_k = probs[k];
                grads[k] += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint i = 0, k = lid; k < K; ++i, k += WG_SIZE) {
              Float noise = PHI_RANDN(rseed);
              Float phi_k = pi_a[k] * phi_sum;
              phi_vec[k] =
                  FABS(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                       SQRT(eps_t * phi_k) * noise);
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
            GLOBAL void* vrand) {
          LOCAL_DECLARE Float grads[K];
          LOCAL_DECLARE Float probs[K];
          LOCAL_DECLARE Float pi_a[K];
          LOCAL_DECLARE Float aux[WG_SIZE];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux, grads, probs, pi_a);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Float* g_phi,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          uint lid = GET_LOCAL_ID();
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
            for (uint k = lid; k < K; k += WG_SIZE) {
              pi[k] = phi[k];
            }
            BARRIER_GLOBAL;
            WG_NORMALIZE_Float(pi, aux, K);
            if (lid == 0) g_phi[n] = aux[0];
          }
        }

        )%%";
#elif 1
// grads/pi_a in shared memory, probs in registers.
// Optimal block size for Titan X = 128
// Note: tried moving more arrays in registers, worked but was slower. Using
// too many registers limits number of concurrent blocks on a single SM.
const std::string kSourcePhiWg =
    mcmc::algorithm::WorkGroupNormalizeProgram(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() +
    "\n"
    "#define WG_SUM_Float_LOCAL_ WG_SUM_" +
    type_name<Float>() +
    "_LOCAL_\n"
    "#define WG_SUML_Float WG_SUML_" +
    type_name<Float>() +
    "\n"
    "#define WG_NORMALIZE_Float WG_NORMALIZE_" +
    type_name<Float>() + "\n" + random::GetRandomHeader() + R"%%(

        #define K_PER_THREAD ((K/WG_SIZE) + (K % WG_SIZE? 1 : 0))

        #define INIT_ARRAYS(i, kk) \
        {\
          uint k = kk;\
          if (k < K) { \
            grads[k] = 0; \
            pi_a[k] = pi[k]; \
          }\
        }
        #define CALC_PROBS(i, kk) \
        {\
          uint k = kk;\
          if (k < K) { \
            Float beta_k = Beta(beta, k); \
            Float f = (y == 1) ? (beta_k - EPSILON) \
                               : (EPSILON - beta_k); \
            Float pi_k = pi_a[k]; \
            Float pin_k = pi_neighbor[k]; \
            Float probs_k = pi_k * (pin_k * f + e); \
            probs_sum += probs_k; \
            probs[i] = probs_k; \
          }\
        }
        #define CALC_GRADS(i, kk) \
        {\
          uint k = kk;\
          if (k < K) { \
            Float pi_k = pi_a[k]; \
            Float probs_k = probs[i]; \
            grads[k] += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum; \
          }\
        }
        #define CALC_PHI(i, kk) \
        {\
          uint k = kk;\
          if (k < K) { \
            Float noise = PHI_RANDN(rseed); \
            Float phi_k = pi_a[k] * phi_sum; \
            phi_vec[k] = \
                FABS(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) + \
                     SQRT(eps_t * phi_k) * noise); \
          }\
        }

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Float* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux,
                                   LOCAL Float* grads,
                                   LOCAL Float *pi_a
                                   ) {
          Float probs[K_PER_THREAD];
          GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          {
            GENERATE_INIT_ARRAYS
            // eg:
            // INIT_ARRAYS(0, lid + 0 * WG_SIZE);
            // INIT_ARRAYS(1, lid + 1 * WG_SIZE);
            // INIT_ARRAYS(2, lid + 2 * WG_SIZE);
            // INIT_ARRAYS(3, lid + 3 * WG_SIZE);
          }

          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Float* pi_neighbor =
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            Float probs_sum = 0;
            // probs
            {
              GENERATE_CALC_PROBS
              // eg:
              // CALC_PROBS(0, lid + 0 * WG_SIZE);
              // CALC_PROBS(1, lid + 1 * WG_SIZE);
              // CALC_PROBS(2, lid + 2 * WG_SIZE);
              // CALC_PROBS(3, lid + 3 * WG_SIZE);
            }
            aux[lid] = probs_sum;
            BARRIER_LOCAL;
            WG_SUM_Float_LOCAL_(aux, K);
            probs_sum = aux[0];
            BARRIER_LOCAL;
            {
              GENERATE_CALC_GRADS
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          {
            GENERATE_CALC_PHI
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
            GLOBAL void* vrand) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          LOCAL_DECLARE Float grads[K];
          LOCAL_DECLARE Float pi_a[K];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux, grads, pi_a);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        KERNEL void update_pi(GLOBAL void* g_pi, GLOBAL Float* g_phi_vec,
                              GLOBAL Float* g_phi,
                              GLOBAL Vertex* mini_batch_nodes,
                              uint num_mini_batch_nodes) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          uint lid = GET_LOCAL_ID();
          for (; i < num_mini_batch_nodes; i += gsize) {
            Vertex n = mini_batch_nodes[i];
            GLOBAL Float* pi = FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, n);
            GLOBAL Float* phi = g_phi_vec + i * K;
            for (uint k = lid; k < K; k += WG_SIZE) {
              pi[k] = phi[k];
            }
            BARRIER_GLOBAL;
            WG_NORMALIZE_Float(pi, aux, K);
            if (lid == 0) g_phi[n] = aux[0];
          }
        }

        )%%";
#endif

PhiUpdater::PhiUpdater(Mode mode, const Config& cfg, clcuda::Queue queue,
                       clcuda::Buffer<Float>& beta,
                       RowPartitionedMatrix<Float>* pi,
                       clcuda::Buffer<Float>& phi, OpenClSet* trainingSet,
                       const std::vector<std::string>& compileFlags,
                       const std::string& baseFuncs)
    : mode_(mode),
      queue_(queue),
      beta_(beta),
      pi_(pi),
      phi_(phi),
      phi_vec(queue_.GetContext(), 2 * cfg.mini_batch_size * cfg.K),
      trainingSet_(trainingSet),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(
          2 * cfg.mini_batch_size *
              (mode == NODE_PER_THREAD ? 1 : cfg.phi_wg_size),
          random::random_seed_t{cfg.phi_seed[0], cfg.phi_seed[1]})),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.phi_wg_size),
      t_update_phi_(0),
      t_update_pi_(0) {
  std::string src;
  switch (mode_) {
    case NODE_PER_THREAD:
      src = kSourcePhi;
      grads_.reset(new clcuda::Buffer<Float>(queue_.GetContext(),
                                             2 * cfg.mini_batch_size * cfg.K));
      probs_.reset(new clcuda::Buffer<Float>(queue_.GetContext(),
                                             2 * cfg.mini_batch_size * cfg.K));
      break;
    case NODE_PER_WORKGROUP: {
      src = kSourcePhiWg;
      uint32_t k_per_thread = k_ / local_ + (k_ % local_ ? 1 : 0);
      for (auto s : std::vector<std::string>{"INIT_ARRAYS", "CALC_PROBS",
                                             "CALC_GRADS",  "CALC_PHI"}) {
        std::ostringstream out;
        for (uint i = 0; i < k_per_thread; ++i) {
          // INIT_ARRAYS(0, lid + 0 * WG_SIZE);
          out << s << "(" << i << ", lid + " << i << " * WG_SIZE);\n";
        }
        src = boost::replace_all_copy(
            src, std::string("GENERATE_") + s, out.str());
      }
    } break;
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
      << std::endl << "#define FloatRowPartitionedMatrix " << type_name<Float>()
      << "RowPartitionedMatrix\n"
      << "#define FloatRowPartitionedMatrix_Row " << type_name<Float>()
      << "RowPartitionedMatrix_Row\n" << src << std::endl;
  prog_.reset(new clcuda::Program(queue_.GetContext(), out.str()));
  std::vector<std::string> opts =
      ::mcmc::GetClFlags(mode == NODE_PER_WORKGROUP ? local_ : 0);
  opts.insert(opts.end(), compileFlags.begin(), compileFlags.end());
  clcuda::BuildStatus status = prog_->Build(queue_.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_->GetBuildInfo(queue_.GetDevice());
  LOG(INFO) << "####################### PHI LOG:" << std::endl
            << prog_->GetBuildInfo(queue_.GetDevice());
  phi_kernel_.reset(new clcuda::Kernel(*prog_, "update_phi"));
  uint32_t arg = 0;
  phi_kernel_->SetArgument(arg++, beta_);
  phi_kernel_->SetArgument(arg++, pi_->Get());
  phi_kernel_->SetArgument(arg++, phi_);
  phi_kernel_->SetArgument(arg++, phi_vec);
  phi_kernel_->SetArgument(arg++, trainingSet->Get()());
  arg += 4;
  if (mode == NODE_PER_THREAD) {
    phi_kernel_->SetArgument(arg++, *grads_);
    phi_kernel_->SetArgument(arg++, *probs_);
  }
  phi_kernel_->SetArgument(arg++, rand_->Get());

  pi_kernel_.reset(new clcuda::Kernel(*prog_, "update_pi"));
  pi_kernel_->SetArgument(0, pi_->Get());
  pi_kernel_->SetArgument(1, phi_vec);
  pi_kernel_->SetArgument(2, phi_);
}

void PhiUpdater::operator()(
    clcuda::Buffer<Vertex>& mini_batch_nodes,  // [X <= 2*MINI_BATCH_SIZE]
    clcuda::Buffer<Vertex>& neighbors,  // [MINI_BATCH_NODES, NUM_NEIGHBORS]
    uint32_t num_mini_batch_nodes) {
  LOG_IF(FATAL, num_mini_batch_nodes == 0) << "mini-batch nodes size = 0!";
  if (grads_ != nullptr) {
    LOG_IF(FATAL, grads_->GetSize() / sizeof(Float) < num_mini_batch_nodes * k_)
        << "grads too small";
    LOG_IF(FATAL, probs_->GetSize() / sizeof(Float) < num_mini_batch_nodes * k_)
        << "probs too small";
  }
  ++count_calls_;
  uint32_t global;
  if (mode_ == NODE_PER_THREAD) {
    global = (num_mini_batch_nodes / local_ +
              (num_mini_batch_nodes % local_ ? 1 : 0));
  } else {
    global = num_mini_batch_nodes;
  }
  global = std::min(global, GetMaxGroups()) * local_;
  LOG_IF(FATAL,
         rand_->GetSeeds().GetSize() / sizeof(random::random_seed_t) < global)
      << "Num seeds smaller than global threads";
  phi_kernel_->SetArgument(5, mini_batch_nodes);
  phi_kernel_->SetArgument(6, neighbors);
  phi_kernel_->SetArgument(7, num_mini_batch_nodes);
  phi_kernel_->SetArgument(8, count_calls_);
  phi_kernel_->Launch(queue_, {global}, {local_}, phi_event_);
  queue_.Finish();
  t_update_phi_ += phi_event_.GetElapsedTime();
  pi_kernel_->SetArgument(3, mini_batch_nodes);
  pi_kernel_->SetArgument(4, num_mini_batch_nodes);
  pi_kernel_->Launch(queue_, {global}, {local_}, pi_event_);
  queue_.Finish();
  t_update_pi_ += pi_event_.GetElapsedTime();
}

bool PhiUpdater::Serialize(std::ostream* out) {
  PhiProperties props;
  props.set_count_calls(count_calls_);
  props.set_update_phi_time(t_update_phi_);
  props.set_update_pi_time(t_update_pi_);
  return (rand_->Serialize(out) && ::mcmc::SerializeMessage(out, props));
}

bool PhiUpdater::Parse(std::istream* in) {
  PhiProperties props;
  if (rand_->Parse(in) && ::mcmc::ParseMessage(in, &props)) {
    count_calls_ = props.count_calls();
    t_update_phi_ = props.update_phi_time();
    t_update_pi_ = props.update_pi_time();
    return true;
  } else {
    return false;
  }
}

}  // namespace mcmc
