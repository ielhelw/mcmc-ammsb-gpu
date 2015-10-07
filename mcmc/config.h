#ifndef __MCMC_CONFIG_H__
#define __MCMC_CONFIG_H__

#include <limits>
#include <memory>
#include <vector>

#include "mcmc/data.h"

namespace mcmc {

struct Config {
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

  uint32_t ppx_wg_size;
};

std::ostream& operator<<(std::ostream& out, const Config& cfg);

std::string MakeCompileFlags(const Config& cfg);

const std::string& GetSourceGuard();

}  // namespace mcmc

#endif  // __MCMC_CONFIG_H__
