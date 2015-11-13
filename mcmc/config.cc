#include "mcmc/config.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>
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

std::string float_to_string(Float f) {
  std::ostringstream out;
  out << std::scientific << f;
  if (type_name<Float>() == "float") {
    out << "f";
  }
  return out.str();
}

std::vector<std::string> MakeCompileFlags(const Config& cfg) {
  std::vector<std::string> ret;
  ret.push_back(std::string("-DFLOAT_TYPE=") + type_name<Float>());
  ret.push_back("-DVERTEX_TYPE=uint");
  ret.push_back("-DEDGE_TYPE=ulong");
  ret.push_back(std::string("-DK=") + std::to_string(cfg.K));
  ret.push_back(std::string("-DN=") + std::to_string(cfg.N));
  ret.push_back(std::string("-DE=") + std::to_string(cfg.E));
  ret.push_back(std::string("-DALPHA=") + float_to_string(cfg.alpha));
  ret.push_back(std::string("-DEPS_A=") + float_to_string(cfg.a));
  ret.push_back(std::string("-DEPS_B=") + float_to_string(cfg.b));
  ret.push_back(std::string("-DEPS_C=") + float_to_string(cfg.c));
  ret.push_back(std::string("-DEPSILON=") + float_to_string(cfg.epsilon));
  ret.push_back(std::string("-DETA0=") + float_to_string(cfg.eta0));
  ret.push_back(std::string("-DETA1=") + float_to_string(cfg.eta1));
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
  out << "ppx-wg: " << cfg.ppx_wg_size << std::endl;
  out << "phi-wg: " << cfg.phi_wg_size << std::endl;
  out << "beta-wg: " << cfg.beta_wg_size << std::endl;
  out << "phi-seed: " << cfg.phi_seed << std::endl;
  out << "beta-seed: " << cfg.beta_seed << std::endl;
  out << "neighbor-seed: " << cfg.neighbor_seed << std::endl;
  out << "|N|: " << cfg.N << std::endl;
  out << "|E|: " << cfg.E << std::endl;
  out << "phi_mode: " << to_string(cfg.phi_mode) << std::endl;
  out << "phi_vwidth: " << cfg.phi_vector_width << std::endl;
  if (cfg.phi_mode == PHI_NODE_PER_WORKGROUP_CODE_GEN) {
    out << "phi_probs_shared: " << cfg.phi_probs_shared << std::endl;
    out << "phi_grads_shared: " << cfg.phi_grads_shared << std::endl;
    out << "phi_pi_shared: " << cfg.phi_pi_shared << std::endl;
  }
  if (cfg.training)
    out << "|Training edges|: " << cfg.training->Size() << std::endl;
  if (cfg.heldout)
    out << "|Heldout edges|: " << cfg.heldout->Size() << std::endl;
  return out;
}

std::istream& operator>>(std::istream& in, PhiUpdaterMode& mode) {
  std::string token;
  in >> token;
  if (boost::iequals(token, "THREAD")) {
    mode = PHI_NODE_PER_THREAD;
  } else if (boost::iequals(token, "WG-NAIVE")) {
    mode = PHI_NODE_PER_WORKGROUP_NAIVE;
  } else if (boost::iequals(token, "WG-SHARED")) {
    mode = PHI_NODE_PER_WORKGROUP_SHARED;
  } else if (boost::iequals(token, "WG-GEN")) {
    mode = PHI_NODE_PER_WORKGROUP_CODE_GEN;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value,
        "Invalid phi mode");
  }
  return in;
}

std::string to_string(const PhiUpdaterMode& mode) {
  switch (mode) {
    case PHI_NODE_PER_THREAD:
      return "THREAD";
    case PHI_NODE_PER_WORKGROUP_NAIVE:
      return "WG-NAIVE";
    case PHI_NODE_PER_WORKGROUP_SHARED:
      return "WG-SHARED";
    case PHI_NODE_PER_WORKGROUP_CODE_GEN:
      return "WG-GEN";
    default:
      LOG(FATAL) << "Invalid phi mode";
  }
  return "";
}

}  // namespace mcmc
