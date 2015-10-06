#include "mcmc/learner.h"

#include <algorithm>
#include <glog/logging.h>
#include <random>

#include "mcmc/algorithm/sum.h"

namespace mcmc {

const std::string kSourceGuards = std::string(
    "#ifndef FLOAT_TYPE \n"
    "#  error \"FLOAT_TYPE is required\" \n"
    "#endif \n"
    "#ifndef VERTEX_TYPE \n"
    "#  error \"VERTEX_TYPE is required\" \n"
    "#endif \n"
    "#ifndef EDGE_TYPE \n"
    "#  error \"EDGE_TYPE is required\" \n"
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
    "#define EPS_A ((Float)EPS_A_) \n"
    "#define EPS_B ((Float)EPS_B_) \n"
    "#define EPS_C ((Float)EPS_C_) \n"
    "#define ETA0 ((Float)ETA0_) \n"
    "#define ETA1 ((Float)ETA1_) \n"
    "#define EPSILON ((Float)EPSILON_) \n");

const std::string kSourceBaseFuncs =
    kSourceGuards + "\n" +
    kClSetHeader + "\n" +
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        typedef FLOAT_TYPE Float; typedef VERTEX_TYPE Vertex;
        typedef EDGE_TYPE Edge;

        inline Vertex Vertex0(Edge e) {
          return (Vertex)((e & 0xffff0000) >> 32);
        }

            inline Vertex Vertex1(Edge e) { return (Vertex)((e & 0x0000ffff)); }

            inline __global Float *
            Pi(__global Float * pi, Vertex u) { return pi + u * K; }

            inline Float get_eps_t(uint step_count) {
          return EPS_A * pow(1 + step_count / EPS_B, -EPS_C);
        }

        );

std::string MakeCompileFlags(const Config& cfg) {
  std::ostringstream out;
  out << "-DFLOAT_TYPE="
      << "float"
      << " "
      << "-DVERTEX_TYPE="
      << "uint"
      << " "
      << "-DEDGE_TYPE="
      << "ulong"
      << " "
      << "-DK=" << cfg.K << " "
      << "-DN=" << cfg.N << " "
      << "-DE=" << cfg.E << " "
      << "-DEPS_A_=" << cfg.a << " "
      << "-DEPS_B_=" << cfg.b << " "
      << "-DEPS_C_=" << cfg.c << " "
      << "-DEPSILON_=" << cfg.epsilon << " "
      << "-DETA0_=" << cfg.eta0 << " "
      << "-DETA1_=" << cfg.eta1 << " ";
  return out.str();
}

Learner::Learner(const Config& cfg, compute::command_queue queue)
    : cfg_(cfg),
      queue_(queue),
      beta_(cfg_.K, queue_.get_context()),
      theta_(2 * cfg_.K, queue_.get_context()),
      pi_(cfg_.N * cfg_.K, queue_.get_context()),
      phi_(cfg_.N * cfg_.K, queue_.get_context()),
      scratch_(queue_.get_context(), cfg_.N * cfg_.K * sizeof(Float)),
      setFactory_(OpenClSetFactory::New(queue_)),
      trainingSet_(setFactory_->CreateSet(cfg_.training->Serialize())),
      heldoutSet_(setFactory_->CreateSet(cfg_.heldout->Serialize())),
      trainingEdges_(cfg_.training_edges.begin(), cfg_.training_edges.end(),
                     queue_),
      heldoutEdges_(cfg_.heldout_edges.begin(), cfg_.heldout_edges.end(),
                    queue_),
      compileFlags_(MakeCompileFlags(cfg_)),
      heldoutPerplexity_(PerplexityCalculator::EDGE_PER_WORKGROUP, cfg_, queue_,
                         beta_, pi_, heldoutEdges_, heldoutSet_.get(),
                         compileFlags_, kSourceBaseFuncs) {
  // gamma generator
  std::mt19937 mt19937;
  std::gamma_distribution<Float> gamma_distribution(cfg_.eta0, cfg_.eta1);
  auto gamma = std::bind(gamma_distribution, mt19937);
  // generate theta
  std::vector<Float> host_theta(theta_.size());
  std::generate(host_theta.begin(), host_theta.end(), gamma);
  compute::copy(host_theta.begin(), host_theta.end(), theta_.begin(), queue_);
  // generate phi
  std::vector<Float> host_phi(phi_.size());
  std::generate(host_phi.begin(), host_phi.end(), gamma);
  compute::copy(host_phi.begin(), host_phi.end(), phi_.begin(), queue_);
  // norm pi
  for (auto i = 0; i < cfg_.N; ++i) {
    Float sum = 0;
    for (auto j = 0; j < cfg_.K; ++j) {
      sum += host_phi[i * cfg_.K + j];
    }
    for (auto j = 0; j < cfg_.K; ++j) {
      host_phi[i * cfg_.K + j] /= sum;
    }
  }
  compute::copy(host_phi.begin(), host_phi.end(), pi_.begin(), queue_);
  // norm beta
    Float sum = 0;
    for (auto j = 0; j < cfg_.K; ++j) {
      sum += host_theta[j];
    }
    for (auto j = 0; j < cfg_.K; ++j) {
      host_theta[j] /= sum;
    }
  compute::copy(host_theta.begin(), host_theta.begin() + cfg_.K, beta_.begin(), queue_);
}

void Learner::run() {
  uint32_t step_count = 1;
  for (; step_count < 10; ++step_count) {
    LOG(INFO) << "ppx[" << step_count << "] = " << heldoutPerplexity_();
  }
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
