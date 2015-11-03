#include "mcmc/config.h"

#include <sstream>

namespace mcmc {

const std::string& GetSourceGuard() {
  static const std::string kSourceGuards = std::string(
      "#ifndef FLOAT_TYPE \n"
      "#  error \"FLOAT_TYPE is required\" \n"
      "#endif \n"
      "#ifndef VERTEX_TYPE \n"
      "#  error \"VERTEX_TYPE is required\" \n"
      "#endif \n"
      "#ifndef EDGE_TYPE \n"
      "#  error \"EDGE_TYPE is required\" \n"
      "#endif \n"
      "#ifndef ALPHA \n"
      "#  error \"ALPHA is required\" \n"
      "#endif \n"
      "#ifndef EPS_A \n"
      "#  error \"EPS_A is required\" \n"
      "#endif \n"
      "#ifndef EPS_B \n"
      "#  error \"EPS_B is required\" \n"
      "#endif \n"
      "#ifndef EPS_C \n"
      "#  error \"EPS_C is required\" \n"
      "#endif \n"
      "#ifndef EPSILON \n"
      "#  error \"EPSILON is required\" \n"
      "#endif \n"
      "#ifndef ETA0 \n"
      "#  error \"ETA0 is required\" \n"
      "#endif \n"
      "#ifndef ETA1 \n"
      "#  error \"ETA1 is required\" \n"
      "#endif \n"
      "#ifndef K \n"
      "#  error \"K is required\" \n"
      "#endif \n"
      "#ifndef N \n"
      "#  error \"N is required\" \n"
      "#endif \n"
      "#ifndef E \n"
      "#  error \"E is required\" \n"
      "#endif \n"
      "#ifndef NUM_NEIGHBORS \n"
      "#  error \"NUM_NEIGHBORS is required\" \n"
      "#endif \n"
      );
  return kSourceGuards;
}

std::vector<std::string> MakeCompileFlags(const Config& cfg) {
  std::string float_suffix = (type_name<Float>() == "float" ? "f" : "");
  std::vector<std::string> ret;
  ret.push_back(std::string("-DFLOAT_TYPE=") + type_name<Float>());
  ret.push_back("-DVERTEX_TYPE=uint");
  ret.push_back("-DEDGE_TYPE=ulong");
  ret.push_back(std::string("-DK=") + std::to_string(cfg.K));
  ret.push_back(std::string("-DN=") + std::to_string(cfg.N));
  ret.push_back(std::string("-DE=") + std::to_string(cfg.E));
  ret.push_back(std::string("-DALPHA=") + std::to_string(cfg.alpha) + float_suffix);
  ret.push_back(std::string("-DEPS_A=") + std::to_string(cfg.a) + float_suffix);
  ret.push_back(std::string("-DEPS_B=") + std::to_string(cfg.b) + float_suffix);
  ret.push_back(std::string("-DEPS_C=") + std::to_string(cfg.c) + float_suffix);
  ret.push_back(std::string("-DEPSILON=") + std::to_string(cfg.epsilon) + float_suffix);
  ret.push_back(std::string("-DETA0=") + std::to_string(cfg.eta0) + float_suffix);
  ret.push_back(std::string("-DETA1=") + std::to_string(cfg.eta1) + float_suffix);
  ret.push_back(std::string("-DNUM_NEIGHBORS=") + std::to_string(cfg.num_node_sample));
  return ret;
}

std::ostream& operator<<(std::ostream& out, const Config& cfg) {
  out << "Config:" << std::endl;
  out << "heldout ratio: " << cfg.heldout_ratio << std::endl;
  out << "alpha: " << cfg.alpha << std::endl;
  out << "a: " << cfg.a << ", b: " << cfg.b << ", c: " << cfg.c << std::endl;
  out << "epsilon: " << cfg.epsilon << std::endl;
  out << "eta: (" << cfg.eta0 << ", " << cfg.eta1 << ")" << std::endl;
  out << "K: " << cfg.K << std::endl;
  out << "m: " << cfg.mini_batch_size << std::endl;
  out << "n: " << cfg.num_node_sample << std::endl;
  out << "strategy: " << to_string(cfg.strategy) << std::endl;
  out << "ppx-wg: : " << cfg.ppx_wg_size << std::endl;
  out << "phi-wg: : " << cfg.phi_wg_size << std::endl;
  out << "beta-wg: : " << cfg.beta_wg_size << std::endl;
  out << "phi-seed: : " << cfg.phi_seed << std::endl;
  out << "beta-seed: : " << cfg.beta_seed << std::endl;
  out << "neighbor-seed: : " << cfg.neighbor_seed << std::endl;
  out << "|N|: " << cfg.N << std::endl;
  out << "|E|: " << cfg.E << std::endl;
  if (cfg.training)
    out << "|Training edges|: " << cfg.training->Size() << std::endl;
  if (cfg.heldout)
    out << "|Heldout edges|: " << cfg.heldout->Size() << std::endl;
  return out;
}

}  // namespace mcmc
