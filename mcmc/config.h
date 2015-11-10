#ifndef __MCMC_CONFIG_H__
#define __MCMC_CONFIG_H__

#include <limits>
#include <memory>
#include <vector>

#include "mcmc/data.h"
#include "mcmc/sample.h"
#include "mcmc/random.h"

namespace mcmc {

enum PhiUpdaterMode {
  PHI_NODE_PER_THREAD,
  PHI_NODE_PER_WORKGROUP_NAIVE,
  PHI_NODE_PER_WORKGROUP_SHARED,
  PHI_NODE_PER_WORKGROUP_CODE_GEN
};

std::istream& operator>>(std::istream& in, PhiUpdaterMode& mode);

std::string to_string(const PhiUpdaterMode& mode);

struct Config {
#ifdef MCMC_CALC_TRAIN_PPX
  Float training_ppx_ratio;  // subset of training edges used for training ppx
                             // calculation
#endif
  Float heldout_ratio;
  Float alpha;
  Float a, b, c;
  Float epsilon;
  Float eta0, eta1;
  uint64_t K;
  uint64_t mini_batch_size;
  uint64_t num_node_sample;
  uint64_t N;
  uint64_t E;
  std::vector<Edge> training_edges;
  std::vector<Edge> heldout_edges;
  std::unique_ptr<mcmc::Set> training;
  std::unique_ptr<mcmc::Set> heldout;
  std::unique_ptr<mcmc::Graph> trainingGraph;
  std::unique_ptr<mcmc::Graph> heldoutGraph;

  uint32_t ppx_wg_size;
  uint32_t ppx_interval;
  uint32_t neighbor_sampler_wg_size;
  uint32_t phi_wg_size;
  uint32_t beta_wg_size;

  ulong2 phi_seed;
  ulong2 beta_seed;
  ulong2 neighbor_seed;

  bool phi_disable_noise;

  SampleStrategy strategy;

  PhiUpdaterMode phi_mode;

  bool phi_probs_shared;
  bool phi_grads_shared;
  bool phi_pi_shared;
  uint32_t phi_vector_width;

  Config() {
#ifdef MCMC_CALC_TRAIN_PPX
    training_ppx_ratio = 0.01;
#endif
    heldout_ratio = 0.01;
    alpha = 0.001;
    a = 0.0315;
    b = 1024;
    c = 0.5;
    epsilon = 1e-7;
    eta0 = 1;
    eta1 = 1;
    K = 32;
    mini_batch_size = 32;
    num_node_sample = 32;
    ppx_wg_size = 32;
    ppx_interval = 100;
    neighbor_sampler_wg_size = 32;
    phi_wg_size = 32;
    beta_wg_size = 32;
    phi_disable_noise = false;
    phi_seed = {42, 43};
    beta_seed = {113, 117};
    neighbor_seed = {3337, 54351};
    strategy = Node;
    phi_mode = PHI_NODE_PER_WORKGROUP_NAIVE;
    phi_probs_shared = true;
    phi_grads_shared = true;
    phi_pi_shared = true;
    phi_vector_width = 1;
  }
};

std::ostream& operator<<(std::ostream& out, const Config& cfg);

std::vector<std::string> MakeCompileFlags(const Config& cfg);

const std::string& GetSourceGuard();

}  // namespace mcmc

#endif  // __MCMC_CONFIG_H__
