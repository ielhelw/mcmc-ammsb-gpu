#include "mcmc/learner.h"

namespace mcmc {

Learner::Learner(const Config& cfg, compute::command_queue queue)
    : cfg_(cfg),
      queue_(queue),
      eta_(queue_.get_context()),
      beta_(cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      pi_(cfg_.N * cfg_.K, queue_.get_context()),
      phi_(cfg_.N * cfg_.K, queue_.get_context()),
      ppx_per_heldout_edge_(cfg_.heldout->Size(), queue_.get_context()),
      ppx_per_heldout_edge_link_likelihood_(cfg_.heldout->Size(),
                                            queue_.get_context()),
      ppx_per_heldout_edge_non_link_likelihood_(cfg_.heldout->Size(),
                                                queue_.get_context()),
      ppx_per_heldout_edge_link_count_(cfg_.heldout->Size(),
                                       queue_.get_context()),
      ppx_per_heldout_edge_non_link_count_(cfg_.heldout->Size(),
                                           queue_.get_context()) {}

void Learner::run() {
  uint32_t step_count = 1;
  for (; step_count < 10; ++step_count) {
  }
}

std::ostream& operator<<(std::ostream& out, const Config& cfg) {
  out << "Config:" << std::endl;
  out << "heldout ratio: " << cfg.heldout_ratio << std::endl;
  out << "alpha: " << cfg.alpha << std::endl;
  out << "a: " << cfg.a << ", b: " << cfg.b << ", c: " << cfg.c << std::endl;
  out << "epsilon: " << cfg.epsilon << std::endl;
  out << "K: " << cfg.K << std::endl;
  out << "m: " << cfg.mini_batch_size << std::endl;
  out << "n: " << cfg.num_node_sample << std::endl;
  out << "|N|: " << cfg.N << std::endl;
  out << "|E|: " << cfg.E << std::endl;
  if (cfg.training)
    out << "|Training edges|: " << cfg.training->Size() << std::endl;
  if (cfg.heldout)
    out << "|Heldout edges|: " << cfg.heldout->Size() << std::endl;
  return out;
}

}  // namespace mcmc
