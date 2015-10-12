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
      "#ifndef ALPHA_ \n"
      "#  error \"ALPHA_ is required\" \n"
      "#endif \n"
      "#ifndef EPS_A_ \n"
      "#  error \"EPS_A_ is required\" \n"
      "#endif \n"
      "#ifndef EPS_B_ \n"
      "#  error \"EPS_B_ is required\" \n"
      "#endif \n"
      "#ifndef EPS_C_ \n"
      "#  error \"EPS_C_ is required\" \n"
      "#endif \n"
      "#ifndef EPSILON_ \n"
      "#  error \"EPSILON_ is required\" \n"
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
      "#define ALPHA ((Float)ALPHA_) \n"
      "#define EPS_A ((Float)EPS_A_) \n"
      "#define EPS_B ((Float)EPS_B_) \n"
      "#define EPS_C ((Float)EPS_C_) \n"
      "#define ETA0 ((Float)ETA0_) \n"
      "#define ETA1 ((Float)ETA1_) \n"
      "#define EPSILON ((Float)EPSILON_) \n");
  return kSourceGuards;
}

std::string MakeCompileFlags(const Config& cfg) {
  std::ostringstream out;
  out << "-DFLOAT_TYPE=" << compute::type_name<Float>() << " "
      << "-DVERTEX_TYPE="
      << "uint"
      << " "
      << "-DEDGE_TYPE="
      << "ulong"
      << " "
      << "-DK=" << cfg.K << " "
      << "-DN=" << cfg.N << " "
      << "-DE=" << cfg.E << " "
      << "-DALPHA_=" << cfg.a << " "
      << "-DEPS_A_=" << cfg.a << " "
      << "-DEPS_B_=" << cfg.b << " "
      << "-DEPS_C_=" << cfg.c << " "
      << "-DEPSILON_=" << cfg.epsilon << " "
      << "-DETA0_=" << cfg.eta0 << " "
      << "-DETA1_=" << cfg.eta1 << " "
      << "-DNUM_NEIGHBORS=" << cfg.num_node_sample << " ";
  return out.str();
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
  out << "|N|: " << cfg.N << std::endl;
  out << "|E|: " << cfg.E << std::endl;
  if (cfg.training)
    out << "|Training edges|: " << cfg.training->Size() << std::endl;
  if (cfg.heldout)
    out << "|Heldout edges|: " << cfg.heldout->Size() << std::endl;
  return out;
}

}  // namespace mcmc
