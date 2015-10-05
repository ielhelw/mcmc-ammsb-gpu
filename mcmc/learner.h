#ifndef __MCMC_LEARNER_H__
#define __MCMC_LEARNER_H__

#include <boost/compute/container/array.hpp>
#include <boost/program_options.hpp>
#include <ostream>

#include "mcmc/data.h"

namespace mcmc {

const std::string kLearnerSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

    typedef double Float; typedef uint Vertex; typedef ulong Edge;

    inline Vertex Vertex0(Edge e) {
      return (Vertex)((e & 0xffff0000) >> 32);
    } inline Vertex Vertex1(Edge e) {
      return (Vertex)((e & 0x0000ffff));
    } inline __global Float *
        Pi(__global Float * pi, Vertex u) { return pi + u * K; }

        inline Float get_eps_t(Float a, Float b, Float c, uint32_t step_count) {
          return a * pow(1 + step_count / b, -c);
        }

        Float calculate_edge_likelihood(__global Float * pi_a,
                                        __global Float * pi_b,
                                        _global Float * beta, bool is_edge) {
          Float s = 0;
          if (is_edge) {
            uint k = 0;
            for (; k < K; ++k) {
              s += pi_a[k] * pi_b[k] * beta[k];
            }
          } else {
            Float sum = 0;
            uint k = 0;
            for (; k < K; ++k) {
              Float f = pi_a[k] * pi_b[k];
              s += f * (1.0 - beta[k]);
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
            __global Float* g_pi, __global Float* g_beta,
            __global Set* edge_set, __global Float* g_ppx_per_edge,
            __global Float* g_link_likelihood,
            __global Float* g_non_link_likelihood, __global uint* g_link_count,
            __global uint* g_non_link_count, uint avg_count, Edge e, ) {
          Vertex u = Vertex0(e);
          Vertex v = Vertex1(e);
          bool is_edge = Set_HasEdge(edge_set, e);
          Float edge_likelihood = calculate_edge_likelihood(
              Pi(g_pi, u), Pi(g_pi, v), g_beta, is_edge);
          Float ppx = *g_ppx_per_edge;
          ppx = (ppx * (avg_count - 1) + edge_likelihood) / avg_count;
          if (is_edge) {
            ++(*g_link_count);
            *g_link_likelihood += log(ppx);
          } else {
            ++(*g_non_link_count);
            *g_non_link_likelihood += log(ppx);
          }
          *g_ppx_per_edge = ppx;
        }

        __kernel void calculate_ppx_partial_for_edge(
            __global Edge* edges, uint num_edges, __global Float* g_pi,
            __global Float* g_beta, __global Set* edge_set,
            __global Float* g_ppx_per_edge, __global Float* g_link_likelihood,
            __global Float* g_non_link_likelihood, __global uint* g_link_count,
            __global uint* g_non_link_count, uint avg_count) {
      size_t i = get_global_id(0);
      for (; i < num_edges; i += get_global_size(0)) {
        calculate_ppx_partial_for_edge(
            g_pi, g_beta, edge_set, g_ppx_per_edge[i], g_link_likelihood[i],
            g_non_link_likelihood[i], g_link_count[i], g_non_link_count[i],
            avg_count, edges[i]);
      }
    }

    );


struct Config {
  Float heldout_ratio;
  Float alpha;
  Float a, b, c;
  Float epsilon;
  uint64_t K;
  uint64_t mini_batch_size;
  uint64_t num_node_sample;
  uint64_t N;
  uint64_t E;
  std::unique_ptr<mcmc::Set> training;
  std::unique_ptr<mcmc::Set> heldout;
};

class Learner {
 public:
  typedef mcmc::Float Float;

  Learner(const Config& cfg, compute::command_queue queue);

  void run();

 private:
  const Config& cfg_;

  compute::command_queue queue_;
  compute::array<Float, 2> eta_;
  compute::vector<Float> beta_;   // [K]
  compute::vector<Float> theta_;  // [K]
  compute::vector<Float> pi_;     // [N,K]
  compute::vector<Float> phi_;    // [N,K]

  compute::vector<Float> ppx_per_heldout_edge_;  // [heldout size]
  compute::vector<Float> ppx_per_heldout_edge_link_likelihood_;
  compute::vector<Float> ppx_per_heldout_edge_non_link_likelihood_;
  compute::vector<compute::uint_> ppx_per_heldout_edge_link_count_;
  compute::vector<compute::uint_> ppx_per_heldout_edge_non_link_count_;
};

std::ostream& operator<<(std::ostream& out, const Config& cfg);

}  // namespace mcmc

#endif  // __MCMC_LEARNER_H__
