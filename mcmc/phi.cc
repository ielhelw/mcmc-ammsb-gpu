#include "mcmc/phi.h"

#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <glog/logging.h>

#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourceVec = R"%%(
        #ifdef VECTOR_WIDTH
          #define Kn (K/VECTOR_WIDTH) 
          #if VECTOR_WIDTH == 2
            #define Floatn Float2
            #define VFABSn vfabs2
            #define VSQRTn vsqrt2
            #define Vn(X) MAKEV2(X)
            #define VLn(X) VL2(X)
            #define VMAXn(X, Y) vmax2(X, Y)
            #ifdef __OPENCL_VERSION__
              #define VBeta() (Float2)(Beta(beta, 2 * k), Beta(beta, 2 * k + 1))
            #else
              #define VBeta() MAKE_FLOAT2(Beta(beta, 2 * k), Beta(beta, 2 * k + 1))
            #endif
            inline Float v_accn(Float2 a) { return a.x + a.y; }
          #elif VECTOR_WIDTH == 4
            #define Floatn Float4
            #define VFABSn vfabs4
            #define VSQRTn vsqrt4
            #define Vn(X) MAKEV4(X)
            #define VLn(X) VL4(X)
            #define VMAXn(X, Y) vmax4(X, Y)
            #ifdef __OPENCL_VERSION__
              #define VBeta() (Float4)(Beta(beta, 4 * k), Beta(beta, 4 * k + 1), Beta(beta, 4 * k + 2), Beta(beta, 4 * k + 3))
            #else
              #define VBeta() MAKE_FLOAT4(Beta(beta, 4 * k), Beta(beta, 4 * k + 1), Beta(beta, 4 * k + 2), Beta(beta, 4 * k + 3))
            #endif
            inline Float v_accn(Float4 a) { return a.x + a.y + a.z + a.w; }
          #elif VECTOR_WIDTH == 8
            #define Floatn Float8
            #define VFABSn vfabs8
            #define VSQRTn vsqrt8
            #define Vn(X) MAKEV8(X)
            #define VLn(X) VL8(X)
            #define VMAXn(X, Y) vmax8(X, Y)
            #define VBeta() (Float8)(Beta(beta, 8 * k), Beta(beta, 8 * k + 1), Beta(beta, 8 * k + 2), Beta(beta, 8 * k + 3), Beta(beta, 8 * k + 4), Beta(beta, 8 * k + 5), Beta(beta, 8 * k + 6), Beta(beta, 8 * k + 7))
            inline Float v_accn(Float8 a) { return a.lo.x + a.lo.y + a.lo.z + a.lo.w + a.hi.x + a.hi.y + a.hi.z + a.hi.w; }
          #elif VECTOR_WIDTH == 16
            #define Floatn Float16
            #define VFABSn vfabs16
            #define VSQRTn vsqrt16
            #define Vn(X) MAKEV16(X)
            #define VLn(X) VL16(X)
            #define VMAXn(X, Y) vmax16(X, Y)
            #define VBeta() (Float16)(Beta(beta, 16 * k), Beta(beta, 16 * k + 1), Beta(beta, 16 * k + 2), Beta(beta, 16 * k + 3), Beta(beta, 16 * k + 4), Beta(beta, 16 * k + 5), Beta(beta, 16 * k + 6), Beta(beta, 16 * k + 7), Beta(beta, 16 * k + 8), Beta(beta, 16 * k + 9), Beta(beta, 16 * k + 10), Beta(beta, 16 * k + 11), Beta(beta, 16 * k + 12), Beta(beta, 16 * k + 13), Beta(beta, 16 * k + 14), Beta(beta, 16 * k + 15))
            inline Float v_accn(Float16 a) { return a.s0 + a.s1 + a.s2 + a.s3 + a.s4 + a.s5 + a.s6 + a.s7 + a.s8 + a.s9 + a.sa + a.sb + a.sc + a.sd + a.se + a.sf; }
          #else
            #error "VECTOR_WIDTH MUST BE 2, 4, 8 or 16"
          #endif
        #else
          #define Kn K
          #define Floatn Float
          #define VFABSn FABS
          #define VSQRTn SQRT
          #define VMAXn(X, Y) MAX(X, Y)
          #define Vn(X) (X)
          #define VLn(X) (X)
          #define VBeta() (Beta(beta, k))
          inline Float v_accn(Float a) { return a; }
        #endif
        #define K_PER_THREAD ((Kn/WG_SIZE) + (Kn % WG_SIZE? 1 : 0))
    )%%";
const std::string kSourcePhi = random::GetRandomHeader() + kSourceVec + R"%%(

        void update_phi_for_node(GLOBAL Float* beta, GLOBAL void* g_pi,
                                 GLOBAL Float* g_phi, GLOBAL Floatn* phi_vec,
                                 GLOBAL Set* edge_set, Vertex node,
                                 GLOBAL Vertex* neighbors, uint step_count,
                                 GLOBAL Floatn* grads,  // K
                                 GLOBAL Floatn* probs,  // K
                                 random_seed_t* rseed) {
          GLOBAL Floatn* pi = (GLOBAL Floatn*) FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          Float phi_sum = g_phi[node];
          for (uint k = 0; k < Kn; ++k) {
            // reset grads
            grads[k] = Vn(0);
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Floatn* pi_neighbor = (GLOBAL Floatn*)
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            Float probs_sum = 0;
            for (uint k = 0; k < Kn; ++k) {
              Floatn beta_k = VBeta();
              Floatn f = (y == 1) ? (beta_k - EPSILON)
                                 : (EPSILON - beta_k);
              Floatn probs_k = pi[k] * (pi_neighbor[k] * f + e);
              probs_sum += v_accn(probs_k);
              probs[k] = probs_k;
            }
            for (uint k = 0; k < Kn; ++k) {
              grads[k] +=
                  (probs[k] / probs_sum) / (pi[k] * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint k = 0; k < Kn; ++k) {
            Floatn noise = VLn(PHI_RANDN(rseed));
            Floatn phi_k = pi[k] * phi_sum;
            Floatn phi_vec_k = 
                VFABSn(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                     VSQRTn(eps_t * phi_k) * noise);
            phi_vec[k] = VMAXn(phi_vec_k, FL(1e-24));
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
              update_phi_for_node(g_beta, g_pi, g_phi, (GLOBAL Floatn*)phi_vec,
                                  (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                  step_count, (GLOBAL Floatn*)grads, (GLOBAL Floatn*)probs, &rseed);
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

const std::string kSourcePiWg = R"%%(
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
// pi_a/probs/grads in thread local memory. Not enough registers, spills.
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
    type_name<Float>() + "\n" + random::GetRandomHeader() + kSourceVec + R"%%(

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Floatn* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux) {
          Floatn grads[K_PER_THREAD];
          Floatn probs[K_PER_THREAD];
          Floatn pi_a[K_PER_THREAD];
          GLOBAL Floatn* pi = (GLOBAL Floatn*) FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
            grads[i] = Vn(0);
            pi_a[i] = pi[k];
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Floatn* pi_neighbor = (GLOBAL Floatn*)
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            // probs
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
                Floatn beta_k = VBeta();
                Floatn f = (y == 1) ? (beta_k - EPSILON)
                                   : (EPSILON - beta_k);
                Floatn pi_k = pi_a[i];
                Floatn pin_k = pi_neighbor[k];
                probs[i] = pi_k * (pin_k * f + e);
            }
            // probs sum
            Float probs_sum = 0;
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
              probs_sum += v_accn(probs[i]);
            }
            aux[lid] = probs_sum;
            BARRIER_LOCAL;
            WG_SUM_Float_LOCAL_(aux, K);
            probs_sum = aux[0];
            BARRIER_LOCAL;
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
                Floatn pi_k = pi_a[i];
                Floatn probs_k = probs[i];
                grads[i] += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
              // create "n" different noise elements
              Floatn noise = VLn(PHI_RANDN(rseed));
              Floatn phi_k = pi_a[i] * phi_sum;
              Floatn phi_vec_k =
                  VFABSn(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[i]) +
                       VSQRTn(eps_t * phi_k) * noise);
              phi_vec[k] = VMAXn(phi_vec_k, FL(1e-24));
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
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, (GLOBAL Floatn*)phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        )%%";
// 25% improvement by placing pi_a/probs/grads in shared memory.
const std::string kSourcePhiWgLMem =
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
    type_name<Float>() + "\n" + random::GetRandomHeader() + kSourceVec + R"%%(

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Floatn* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux,
                                   LOCAL Floatn* grads,
                                   LOCAL Floatn* probs,
                                   LOCAL Floatn* pi_a
                                   ) {
          GLOBAL Floatn* pi = (GLOBAL Floatn*)FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
          Float eps_t = get_eps_t(step_count);
          uint lid = GET_LOCAL_ID();
          // phi sum
          Float phi_sum = g_phi[node];
          // reset grads
          for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
            grads[k] = Vn(0);
            pi_a[k] = pi[k];
          }
          for (uint i = 0; i < NUM_NEIGHBORS; ++i) {
            Vertex neighbor = neighbors[i];
            GLOBAL Floatn* pi_neighbor = (GLOBAL Floatn*)
                FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, neighbor);
            Edge edge = MakeEdge(min(node, neighbor), max(node, neighbor));
            bool y = Set_HasEdge(edge_set, edge);
            Float e = (y == 1 ? EPSILON : FL(1.0) - EPSILON);
            // probs
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
                Floatn beta_k = VBeta();
                Floatn f = (y == 1) ? (beta_k - EPSILON)
                                   : (EPSILON - beta_k);
                Floatn pi_k = pi_a[k];
                Floatn pin_k = pi_neighbor[k];
                probs[k] = pi_k * (pin_k * f + e);
            }
            // probs sum
            Float probs_sum = 0;
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
              probs_sum += v_accn(probs[k]);
            }
            aux[lid] = probs_sum;
            BARRIER_LOCAL;
            WG_SUM_Float_LOCAL_(aux, K);
            probs_sum = aux[0];
            BARRIER_LOCAL;
            for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
                Floatn pi_k = pi_a[k];
                Floatn probs_k = probs[k];
                grads[k] += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum;
            }
          }
          Float Nn = (FL(1.0) * N) / NUM_NEIGHBORS;
          for (uint i = 0, k = lid; k < Kn; ++i, k += WG_SIZE) {
              // create "n" different noise elements
              Floatn noise = VLn(PHI_RANDN(rseed));
              Floatn phi_k = pi_a[k] * phi_sum;
              Floatn phi_vec_k =
                  VFABSn(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * grads[k]) +
                       VSQRTn(eps_t * phi_k) * noise);
              phi_vec[k] = VMAXn(phi_vec_k, FL(1e-24));
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
          LOCAL_DECLARE Floatn grads[Kn];
          LOCAL_DECLARE Floatn probs[Kn];
          LOCAL_DECLARE Floatn pi_a[Kn];
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
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, (GLOBAL Floatn*)phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux, grads, probs, pi_a);
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        )%%";
// grads/pi_a in shared memory, probs in registers.
// Optimal block size for Titan X = 128
const std::string kSourcePhiWgLMemReg =
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
    type_name<Float>() + "\n" + random::GetRandomHeader() + kSourceVec + R"%%(
        #ifdef PROBS_IS_SHARED
          #define PROBS(i, k) probs[k]
        #else
          #define PROBS(i, k) probs[i]
        #endif
        #ifdef GRADS_IS_SHARED
          #define GRADS(i, k) grads[k]
        #else
          #define GRADS(i, k) grads[i]
        #endif
        #ifdef PI_A_IS_SHARED
          #define PI_A(i, k) pi_a[k]
        #else
          #define PI_A(i, k) pi_a[i]
        #endif
        #define INIT_ARRAYS(i, kk) \
        {\
          const uint k = kk;\
          if (k < Kn) { \
            GRADS(i, k) = Vn(FL(0.0)); \
            PI_A(i, k) = pi[k]; \
          }\
        }
        #define CALC_PROBS(i, kk) \
        {\
          const uint k = kk;\
          if (k < Kn) { \
            Floatn beta_k = VBeta();\
            Floatn f = (y == 1) ? (beta_k - EPSILON) \
                                : (EPSILON - beta_k); \
            Floatn pi_k = PI_A(i, k); \
            Floatn pin_k = pi_neighbor[k]; \
            Floatn probs_k = pi_k * (pin_k * f + e); \
            probs_sum += v_accn(probs_k); \
            PROBS(i, k) = probs_k; \
          }\
        }
        #define CALC_GRADS(i, kk) \
        {\
          const uint k = kk;\
          if (k < Kn) { \
            Floatn pi_k = PI_A(i, k); \
            Floatn probs_k = PROBS(i, k); \
            GRADS(i, k) += (probs_k / probs_sum) / (pi_k * phi_sum) - FL(1.0) / phi_sum; \
          }\
        }
        #define CALC_PHI(i, kk) \
        {\
          const uint k = kk;\
          if (k < Kn) { \
            /* create "n" different noise elements */ \
            Floatn noise = VLn(PHI_RANDN(rseed)); \
            Floatn phi_k = (PI_A(i, k) * phi_sum); \
            Floatn phi_vec_k = \
                VFABSn(phi_k + eps_t / 2 * (ALPHA - phi_k + Nn * GRADS(i, k)) + \
                     VSQRTn(eps_t * phi_k) * noise); \
            phi_vec[k] = VMAXn(phi_vec_k, FL(1e-24)); \
          }\
        }

        void update_phi_for_nodeWG(GLOBAL Float* beta, GLOBAL void* g_pi,
                                   GLOBAL Float* g_phi, GLOBAL Floatn* phi_vec,
                                   GLOBAL Set* edge_set, Vertex node,
                                   GLOBAL Vertex* neighbors, uint step_count,
                                   random_seed_t* rseed,
                                   LOCAL Float* aux
#ifdef PROBS_IS_SHARED
                                   , LOCAL Floatn* probs
#endif
#ifdef GRADS_IS_SHARED
                                   , LOCAL Floatn* grads
#endif
#ifdef PI_A_IS_SHARED
                                   , LOCAL Floatn *pi_a
#endif
                                   ) {
#ifndef PROBS_IS_SHARED
          Floatn probs[K_PER_THREAD];
#endif
#ifndef GRADS_IS_SHARED
          Floatn grads[K_PER_THREAD];
#endif
#ifndef PI_A_IS_SHARED
          Floatn pi_a[K_PER_THREAD];
#endif
          GLOBAL Floatn* pi = (GLOBAL Floatn*)FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, node);
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
            GLOBAL Floatn* pi_neighbor = (GLOBAL Floatn*)
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
#ifdef PROBS_IS_SHARED
          LOCAL_DECLARE Floatn probs[Kn];
#endif
#ifdef GRADS_IS_SHARED
          LOCAL_DECLARE Floatn grads[Kn];
#endif
#ifdef PI_A_IS_SHARED
          LOCAL_DECLARE Floatn pi_a[Kn];
#endif
          uint i = GET_GROUP_ID();
          uint gsize = GET_NUM_GROUPS();
          GLOBAL Random* rand = (GLOBAL Random*)vrand;
          if (i < num_mini_batch_nodes) {
            random_seed_t rseed = rand->base_[GET_GLOBAL_ID()];
            for (; i < num_mini_batch_nodes; i += gsize) {
              Vertex node = mini_batch_nodes[i];
              GLOBAL Vertex* node_neighbors = neighbors + i * NUM_NEIGHBORS;
              GLOBAL Float* phi_vec = g_phi_vec + i * K;
              update_phi_for_nodeWG(g_beta, g_pi, g_phi, (GLOBAL Floatn*)phi_vec,
                                    (GLOBAL Set*)training_edge_set, node, node_neighbors,
                                    step_count, &rseed, aux
#ifdef PROBS_IS_SHARED
                                    , probs
#endif
#ifdef GRADS_IS_SHARED
                                    , grads
#endif
#ifdef PI_A_IS_SHARED
                                    , pi_a
#endif
                                    );
            }
            rand->base_[GET_GLOBAL_ID()] = rseed;
          }
        }

        )%%";

PhiUpdater::PhiUpdater(const Config& cfg, clcuda::Queue queue,
                       clcuda::Buffer<Float>& beta,
                       RowPartitionedMatrix<Float>* pi,
                       clcuda::Buffer<Float>& phi, OpenClSet* trainingSet,
                       const std::vector<std::string>& compileFlags,
                       const std::string& baseFuncs)
    : mode_(cfg.phi_mode),
      queue_(queue),
      beta_(beta),
      pi_(pi),
      phi_(phi),
      phi_vec(queue_.GetContext(),
              std::max(2 * cfg.mini_batch_size,
                       1 + cfg.trainingGraph->MaxFanOut()) *
                  cfg.K),
      trainingSet_(trainingSet),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(
          std::max(2 * cfg.mini_batch_size,
                   1 + cfg.trainingGraph->MaxFanOut()) *
              (mode_ == PHI_NODE_PER_THREAD ? 1 : cfg.phi_wg_size),
          random::random_seed_t{cfg.phi_seed[0], cfg.phi_seed[1]})),
      count_calls_(0),
      k_(cfg.K),
      local_(cfg.phi_wg_size),
      t_update_phi_(0),
      t_update_pi_(0) {
  std::string src;
  switch (mode_) {
    case PHI_NODE_PER_THREAD:
      src = kSourcePhi;
      grads_.reset(new clcuda::Buffer<Float>(
          queue_.GetContext(), std::max(2 * cfg.mini_batch_size,
                                        1 + cfg.trainingGraph->MaxFanOut()) *
                                   cfg.K));
      probs_.reset(new clcuda::Buffer<Float>(
          queue_.GetContext(), std::max(2 * cfg.mini_batch_size,
                                        1 + cfg.trainingGraph->MaxFanOut()) *
                                   cfg.K));
      break;
    case PHI_NODE_PER_WORKGROUP_NAIVE:
      src = kSourcePhiWg + kSourcePiWg;
      break;
    case PHI_NODE_PER_WORKGROUP_SHARED:
      src = kSourcePhiWgLMem + kSourcePiWg;
      break;
    case PHI_NODE_PER_WORKGROUP_CODE_GEN: {
      src = kSourcePhiWgLMemReg + kSourcePiWg;
      uint32_t k_per_thread = (k_ / cfg.phi_vector_width) / local_ +
                              ((k_ / cfg.phi_vector_width) % local_ ? 1 : 0);
      for (auto s : std::vector<std::string>{"INIT_ARRAYS", "CALC_PROBS",
                                             "CALC_GRADS",  "CALC_PHI"}) {
        std::ostringstream out;
        for (uint i = 0; i < k_per_thread; ++i) {
          // INIT_ARRAYS(0, lid + 0 * WG_SIZE);
          out << s << "(" << i << ", lid + " << i << " * WG_SIZE);\n";
        }
        src = boost::replace_all_copy(src, std::string("GENERATE_") + s,
                                      out.str());
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
      ::mcmc::GetClFlags(mode_ != PHI_NODE_PER_THREAD ? local_ : 0);
  opts.insert(opts.end(), compileFlags.begin(), compileFlags.end());
  if (cfg.phi_probs_shared) opts.push_back("-DPROBS_IS_SHARED");
  if (cfg.phi_grads_shared) opts.push_back("-DGRADS_IS_SHARED");
  if (cfg.phi_pi_shared) opts.push_back("-DPI_A_IS_SHARED");
  if (cfg.phi_vector_width > 1)
    opts.push_back(std::string("-DVECTOR_WIDTH=") +
                   std::to_string(cfg.phi_vector_width));
#if 0
  {
    std::vector<std::string> strs;
    std::string S = out.str();
    boost::split(strs, S, boost::is_any_of("\n"));
    for (uint32_t i = 0; i < strs.size(); ++i) {
      LOG(INFO) << i << ": " << strs[i];
    }
  }
#endif
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
  if (mode_ == PHI_NODE_PER_THREAD) {
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
  if (mode_ == PHI_NODE_PER_THREAD) {
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
