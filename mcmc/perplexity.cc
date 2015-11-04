#include "mcmc/perplexity.h"

#include <algorithm>
#include <chrono>
#ifdef MCMC_USE_CL
#include <boost/compute/algorithm/reduce.hpp>
#endif
#include <glog/logging.h>
#include "mcmc/algorithm/sum.h"
#include "mcmc/algorithm/normalize.h"

namespace mcmc {

const std::string kSourcePerplexity = R"%%(

    Float calculate_edge_likelihood(GLOBAL Float * pi_a, GLOBAL Float * pi_b,
                                    GLOBAL Float * beta, bool is_edge) {
      Float s = 0;
      if (is_edge) {
        uint k = 0;
        for (; k < K; ++k) {
          s += pi_a[k] * pi_b[k] * Beta(beta, k);
        }
      } else {
        Float sum = 0;
        uint k = 0;
        for (; k < K; ++k) {
          Float f = pi_a[k] * pi_b[k];
          s += f * (FL(1.0) - Beta(beta, k));
          sum += f;
        }
        s += (FL(1.0) - sum) * (FL(1.0) - EPSILON);
      }
      if (s < FL(1.0e-30)) {
        s = FL(1.0e-30);
      }
      return s;
    }

    void calculate_ppx_partial_for_edge_(
        GLOBAL void* g_pi, GLOBAL Float* g_beta, GLOBAL Set* edge_set,
        GLOBAL Float* g_ppx_per_edge, GLOBAL Float* g_link_likelihood,
        GLOBAL Float* g_non_link_likelihood, GLOBAL uint* g_link_count,
        GLOBAL uint* g_non_link_count, uint avg_count, Edge e) {
      Vertex u = Vertex0(e);
      Vertex v = Vertex1(e);
      bool is_edge = Set_HasEdge(edge_set, e);
      Float edge_likelihood = calculate_edge_likelihood(
          FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, u),
          FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, v), g_beta, is_edge);
      Float ppx = *g_ppx_per_edge;
      ppx = (ppx * (avg_count - 1) + edge_likelihood) / avg_count;
      if (is_edge) {
        *g_link_count = 1;
        *g_link_likelihood = LOG(ppx);
        *g_non_link_count = 0;
        *g_non_link_likelihood = 0;
      } else {
        (*g_non_link_count) = 1;
        *g_non_link_likelihood = LOG(ppx);
        *g_link_count = 0;
        *g_link_likelihood = 0;
      }
      *g_ppx_per_edge = ppx;
    }

    KERNEL void calculate_ppx_partial_for_edge(
        GLOBAL Edge* edges, uint num_edges, GLOBAL void* g_pi,
        GLOBAL Float* g_beta, GLOBAL void* /* Set* */ void_edge_set,
        GLOBAL Float* g_ppx_per_edge,         // [num global threads]
        GLOBAL Float* g_link_likelihood,      // [num global threads]
        GLOBAL Float* g_non_link_likelihood,  // [num global threads]
        GLOBAL uint* g_link_count,            // [num global threads]
        GLOBAL uint* g_non_link_count,        // [num global threads]
        uint avg_count) {
      size_t i = GET_GLOBAL_ID();
      for (; i < num_edges; i += GET_GLOBAL_SIZE()) {
        calculate_ppx_partial_for_edge_(
            g_pi, g_beta, (GLOBAL Set*)void_edge_set, g_ppx_per_edge + i,
            g_link_likelihood + i, g_non_link_likelihood + i, g_link_count + i,
            g_non_link_count + i, avg_count, edges[i]);
      }
    }

    )%%";

const std::string kSourcePerplexityWg =
    mcmc::algorithm::WorkGroupSum(type_name<Float>()) + "\n" +
    "#define WG_SUM_Float WG_SUM_" + type_name<Float>() +
    "\n"
    R"%%(

        Float calculate_edge_likelihood_WG(
            GLOBAL Float * pi_a, GLOBAL Float * pi_b, GLOBAL Float * beta,
            bool is_edge, GLOBAL Float* scratch, LOCAL Float* aux) {
          uint lid = GET_LOCAL_ID();
          Float s = 0;
          if (is_edge) {
            for (uint i = lid; i < K; i += GET_LOCAL_SIZE()) {
              scratch[i] = pi_a[i] * pi_b[i] * Beta(beta, i);
            }
            BARRIER_GLOBAL;
            WG_SUM_Float(scratch, aux, K);
            s = aux[0];
          } else {
            Float sum = 0;
            for (uint i = lid; i < K; i += GET_LOCAL_SIZE()) {
              scratch[i] = pi_a[i] * pi_b[i];
            }
            BARRIER_GLOBAL;
            WG_SUM_Float(scratch, aux, K);
            sum = aux[0];
            BARRIER_LOCAL;
            for (uint i = lid; i < K; i += GET_LOCAL_SIZE()) {
              scratch[i] = pi_a[i] * pi_b[i] * (FL(1.0) - Beta(beta, i));
            }
            BARRIER_GLOBAL;
            WG_SUM_Float(scratch, aux, K);
            s = aux[0];
            s += (FL(1.0) - sum) * (FL(1.0) - EPSILON);
          }
          if (s < FL(1.0e-30)) {
            s = FL(1.0e-30);
          }
          return s;
        }

        void calculate_ppx_partial_for_edge_(
            GLOBAL void* g_pi, GLOBAL Float* g_beta, GLOBAL Set* edge_set,
            GLOBAL Float* g_ppx_per_edge, GLOBAL Float* g_link_likelihood,
            GLOBAL Float* g_non_link_likelihood, GLOBAL uint* g_link_count,
            GLOBAL uint* g_non_link_count, uint avg_count, Edge e,
            GLOBAL Float* scratch, LOCAL Float* aux) {
          Vertex u = Vertex0(e);
          Vertex v = Vertex1(e);
          bool is_edge = Set_HasEdge(edge_set, e);
          Float edge_likelihood = calculate_edge_likelihood_WG(
              FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, u),
              FloatRowPartitionedMatrix_Row((GLOBAL FloatRowPartitionedMatrix*)g_pi, v), g_beta, is_edge, scratch,
              aux);
          if (GET_LOCAL_ID() == 0) {
            Float ppx = *g_ppx_per_edge;
            ppx = (ppx * (avg_count - 1) + edge_likelihood) / avg_count;
            if (is_edge) {
              *g_link_count = 1;
              *g_link_likelihood = LOG(ppx);
              *g_non_link_count = 0;
              *g_non_link_likelihood = 0;
            } else {
              *g_non_link_count = 1;
              *g_non_link_likelihood = LOG(ppx);
              *g_link_count = 0;
              *g_link_likelihood = 0;
            }
            *g_ppx_per_edge = ppx;
          }
        }

        KERNEL void calculate_ppx_partial_for_edge(
            GLOBAL Edge* edges, uint num_edges, GLOBAL void* g_pi,
            GLOBAL Float* g_beta, GLOBAL void* /* Set* */ void_edge_set,
            GLOBAL Float* g_ppx_per_edge,         // [num work groups]
            GLOBAL Float* g_link_likelihood,      // [num work groups]
            GLOBAL Float* g_non_link_likelihood,  // [num work groups]
            GLOBAL uint* g_link_count,            // [num work groups]
            GLOBAL uint* g_non_link_count,        // [num work groups]
            uint avg_count,
            GLOBAL Float* scratch  // [num work groups * K]
            ) {
          LOCAL_DECLARE Float aux[WG_SIZE];
          size_t i = GET_GROUP_ID();
          scratch += GET_GROUP_ID() * K;
          for (; i < num_edges; i += GET_NUM_GROUPS()) {
            calculate_ppx_partial_for_edge_(
                g_pi, g_beta, (GLOBAL Set*)void_edge_set, g_ppx_per_edge + i,
                g_link_likelihood + i, g_non_link_likelihood + i,
                g_link_count + i, g_non_link_count + i, avg_count, edges[i],
                scratch, aux);
          }
        }

        )%%";

PerplexityCalculatorBase::PerplexityCalculatorBase(
    Mode mode, const Config& cfg, clcuda::Queue queue,
    clcuda::Buffer<Float>& beta, RowPartitionedMatrix<Float>* pi,
    clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
    const std::vector<std::string>& compileFlags, const std::string& baseFuncs)
    : queue_(queue),
      beta_(beta),
      pi_(pi),
      edges_(edges),
      edgeSet_(edgeSet),
      ppx_per_edge_(queue_.GetContext(), edges_.GetSize() / sizeof(Edge)),
      link_count_(1),
      non_link_count_(1),
      link_likelihood_(1),
      non_link_likelihood_(1),
      count_calls_(0),
      local_(cfg.ppx_wg_size),
      t_ppx_(0), t_accumulate_(0) {
  std::vector<Float> ppx_init(ppx_per_edge_.GetSize() / sizeof(Float), 0);
  ppx_per_edge_.Write(queue_, ppx_init.size(), ppx_init);
  const std::string* src = NULL;
  switch (mode) {
    case EDGE_PER_THREAD:
      src = &kSourcePerplexity;
      global_ = ((edges_.GetSize() / sizeof(Edge)) / local_ +
                 ((edges_.GetSize() / sizeof(Edge)) % local_ ? 1 : 0));
      break;
    case EDGE_PER_WORKGROUP:
      src = &kSourcePerplexityWg;
      global_ = (edges.GetSize() / sizeof(Edge));
      break;
    default:
      LOG(FATAL) << "Cannot recognize mode: " << mode;
  }
  global_ = std::min(global_, GetMaxGroups()) * local_;
  prog_.reset(new clcuda::Program(
      queue_.GetContext(),
      baseFuncs + GetRowPartitionedMatrixHeader<Float>() +
          "#define FloatRowPartitionedMatrix " + type_name<Float>() +
          "RowPartitionedMatrix\n"
          "#define FloatRowPartitionedMatrix_Row " +
          type_name<Float>() + "RowPartitionedMatrix_Row\n" + *src));
  std::vector<std::string> opts =
      ::mcmc::GetClFlags(mode == EDGE_PER_WORKGROUP ? local_ : 0);
  opts.insert(opts.end(), compileFlags.begin(), compileFlags.end());
  clcuda::BuildStatus status = prog_->Build(queue_.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_->GetBuildInfo(queue_.GetDevice());
  LOG(INFO) << "####################### PERPLEXITY LOG:" << std::endl
            << prog_->GetBuildInfo(queue_.GetDevice());
  // ppx_kernel
  kernel_.reset(new clcuda::Kernel(*prog_, "calculate_ppx_partial_for_edge"));
  kernel_->SetArgument(0, edges_);
  kernel_->SetArgument(1,
                       static_cast<uint32_t>(edges_.GetSize() / sizeof(Edge)));
  kernel_->SetArgument(2, pi_->Get());
  kernel_->SetArgument(3, beta_);
  kernel_->SetArgument(4, edgeSet_->Get()());
  kernel_->SetArgument(5, ppx_per_edge_);
  // kernel_.SetArgument(10, count_calls_);
  if (mode == EDGE_PER_WORKGROUP) {
    scratch_.reset(new clcuda::Buffer<Float>(
        queue_.GetContext(), (edges_.GetSize() / sizeof(Edge)) * cfg.K));
    kernel_->SetArgument(11, *scratch_);
  }
}

Float PerplexityCalculatorBase::operator()() {
  count_calls_++;
  kernel_->SetArgument(10, count_calls_);
  clcuda::Event event;
  kernel_->Launch(queue_, {global_}, {local_}, event);
  queue_.Finish();
  t_ppx_ += event.GetElapsedTime();
  
  auto t1 = std::chrono::high_resolution_clock::now();
  AccumulateVectors();
  auto t2 = std::chrono::high_resolution_clock::now();
  t_accumulate_ +=
    std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t2).count();
  double avg_likelihood = 0.0;
  if (link_count_[0] + non_link_count_[0] != 0) {
    avg_likelihood = (link_likelihood_[0] + non_link_likelihood_[0]) /
                     (link_count_[0] + non_link_count_[0]);
  }
  LOG(INFO) << "link_count " << link_count_[0];
  LOG(INFO) << "non_link_count " << non_link_count_[0];
  LOG(INFO) << "link_likelihood " << link_likelihood_[0];
  LOG(INFO) << "non_link_likelihood " << non_link_likelihood_[0];
  return (-avg_likelihood);
}

bool PerplexityCalculatorBase::Serialize(std::ostream* out) {
  PerplexityProperties props;
  props.set_count_calls(count_calls_);
  props.set_ppx_time(t_ppx_);
  props.set_accumulate_time(t_accumulate_);
  return ::mcmc::SerializeMessage(out, props) &&
         ::mcmc::Serialize(out, &ppx_per_edge_, &queue_);
}

bool PerplexityCalculatorBase::Parse(std::istream* in) {
  PerplexityProperties props;
  if (!::mcmc::ParseMessage(in, &props)) return false;
  count_calls_ = props.count_calls();
  t_ppx_ = props.ppx_time();
  t_accumulate_ = props.accumulate_time();
  if (!::mcmc::Parse(in, &ppx_per_edge_, &queue_)) return false;
  return true;
}

#ifdef MCMC_USE_CL
PerplexityCalculatorCl::PerplexityCalculatorCl(
    Mode mode, const Config& cfg, clcuda::Queue queue,
    clcuda::Buffer<Float>& beta, RowPartitionedMatrix<Float>* pi,
    clcuda::Buffer<Edge>& edges, OpenClSet* edgeSet,
    const std::vector<std::string>& compileFlags, const std::string& baseFuncs)
    : PerplexityCalculatorBase(mode, cfg, queue, beta, pi, edges, edgeSet,
                               compileFlags, baseFuncs),
      compute_queue_(queue_(), true),
      ppx_per_edge_link_likelihood_(edges_.GetSize() / sizeof(Edge), 0,
                                    compute_queue_),
      ppx_per_edge_non_link_likelihood_(edges_.GetSize() / sizeof(Edge), 0,
                                        compute_queue_),
      ppx_per_edge_link_count_(edges_.GetSize() / sizeof(Edge), 0,
                               compute_queue_),
      ppx_per_edge_non_link_count_(edges_.GetSize() / sizeof(Edge), 0,
                                   compute_queue_) {
  kernel_->SetArgument(6, ppx_per_edge_link_likelihood_.get_buffer().get());
  kernel_->SetArgument(7, ppx_per_edge_non_link_likelihood_.get_buffer().get());
  kernel_->SetArgument(8, ppx_per_edge_link_count_.get_buffer().get());
  kernel_->SetArgument(9, ppx_per_edge_non_link_count_.get_buffer().get());
}

void PerplexityCalculatorCl::AccumulateVectors() {
  boost::compute::reduce(ppx_per_edge_link_count_.begin(),
                         ppx_per_edge_link_count_.end(), link_count_.begin(),
                         compute_queue_);
  boost::compute::reduce(ppx_per_edge_non_link_count_.begin(),
                         ppx_per_edge_non_link_count_.end(),
                         non_link_count_.begin(), compute_queue_);
  boost::compute::reduce(ppx_per_edge_link_likelihood_.begin(),
                         ppx_per_edge_link_likelihood_.end(),
                         link_likelihood_.begin(), compute_queue_);
  boost::compute::reduce(ppx_per_edge_non_link_likelihood_.begin(),
                         ppx_per_edge_non_link_likelihood_.end(),
                         non_link_likelihood_.begin(), compute_queue_);
}
#endif

}  // namespace mcmc
