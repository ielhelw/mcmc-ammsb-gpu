#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "mcmc/learner.h"
#include "mcmc/data.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace clcuda = mcmc::clcuda;

clcuda::Device ChooseDevice() {
  clcuda::Platform platform((size_t)0);
  return clcuda::Device(platform, 0);
}

sig_atomic_t signaled = 0;

void handler(int sig __attribute((unused))) { signaled = 1; }

int main(int argc, char **argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  {
    std::ostringstream s;
    for (int i = 0; i < argc; ++i) {
      s << argv[i] << " ";
    }
    LOG(INFO) << s.str();
  }
  std::string filename;
  mcmc::Config cfg;
  po::options_description options;
  uint32_t max_iters;
  bool dumpDataset = false;
  bool loadDataset = false;
  std::string loadFile, dumpFile;
  options.add_options()
    ("help,h", "Show usage")
    ("file,f", po::value(&filename),
      "Graph data file")
#ifdef MCMC_CALC_TRAIN_PPX
    ("train-ppx-ratio", po::value(&cfg.training_ppx_ratio)->default_value(0.01))
#endif
    ("heldout-ratio,r", po::value(&cfg.heldout_ratio)->default_value(0.01))
    ("alpha", po::value(&cfg.alpha)->default_value(0))
    ("a,a", po::value(&cfg.a)->default_value(0.0315))
    ("b,b", po::value(&cfg.b)->default_value(1024))
    ("c,c", po::value(&cfg.c)->default_value(0.5))
    ("epsilon,e", po::value(&cfg.epsilon)->default_value(1e-7))
    ("eta0", po::value(&cfg.eta0)->default_value(1))
    ("eta1", po::value(&cfg.eta1)->default_value(1))
    ("k,k", po::value(&cfg.K)->default_value(32))
    ("mini_batch,m", po::value(&cfg.mini_batch_size)->default_value(32))
    ("neighbors,n", po::value(&cfg.num_node_sample)->default_value(32))
    ("ppx-wg", po::value(&cfg.ppx_wg_size)->default_value(32))
    ("ppx-interval,i", po::value(&cfg.ppx_interval)->default_value(100))
    ("phi-wg", po::value(&cfg.phi_wg_size)->default_value(32))
    ("beta-wg", po::value(&cfg.beta_wg_size)->default_value(32))
    ("max-iters,x", po::value(&max_iters)->default_value(100))
    ("sample,s", po::value(&cfg.strategy)->default_value(mcmc::Node))
    ("sampler-wg", po::value(&cfg.neighbor_sampler_wg_size)->default_value(32))
    ("phi-seed", po::value(&cfg.phi_seed)->default_value({42, 43}))
    ("beta-seed", po::value(&cfg.beta_seed)->default_value({44, 45}))
    ("neighbor-seed", po::value(&cfg.neighbor_seed)->default_value({56, 57}))
    ("phi-mode", po::value(&cfg.phi_mode)->default_value(mcmc::PHI_NODE_PER_WORKGROUP_NAIVE))
    ("phi-probs-shared", po::value(&cfg.phi_probs_shared)->default_value(true))
    ("phi-grads-shared", po::value(&cfg.phi_grads_shared)->default_value(true))
    ("phi-pi-shared", po::value(&cfg.phi_pi_shared)->default_value(true))
    ("phi-vwidth", po::value(&cfg.phi_vector_width)->default_value(1))
    ("beta-sum-grads-vwidth", po::value(&cfg.sum_grads_vector_width)->default_value(1))
    ("dump-data", po::value(&dumpDataset)->default_value(false))
    ("dump-file", po::value(&dumpFile))
    ("load-data", po::value(&loadDataset)->default_value(false))
    ("load-file", po::value(&loadFile))
  ;
  po::variables_map options_vm;
  po::store(po::parse_command_line(argc, argv, options), options_vm);
  if (options_vm.count("help")) {
    std::cout << options << std::endl;
    return 1;
  }
  po::notify(options_vm);
  LOG_IF(FATAL,
         !loadDataset && (!fs::exists(filename) || fs::is_directory(filename)))
      << "Failed to detect file: " << filename;
  LOG_IF(FATAL, loadDataset && loadFile.empty()) << "load-file is required with load-data";
  LOG_IF(FATAL, dumpDataset && dumpFile.empty()) << "dump-file is required with dump-data";
  clcuda::Device dev = ChooseDevice();
  clcuda::Context context(dev);
  clcuda::Queue queue(context, dev);
  LOG(INFO) << "OpenCL:" << endl << "  Platform: " << dev.Vendor() << endl
            << "  Device: " << dev.Name() << endl
            << "  Device Driver: " << dev.Version();
  std::vector<mcmc::Edge> unique_edges;
  if (!loadDataset) {
    if (!mcmc::GetUniqueEdgesFromFile(filename, &cfg.N, &unique_edges) ||
        !mcmc::GenerateSetsFromEdges(cfg.N, unique_edges, cfg.heldout_ratio,
                                     &cfg.training_edges, &cfg.heldout_edges,
                                     &cfg.training, &cfg.heldout)) {
      LOG(FATAL) << "Failed to generate sets from file " << filename;
    }
    size_t num_edges = unique_edges.size();
    if (dumpDataset) {
      std::ofstream out_file(dumpFile, std::ofstream::binary);
      boost::iostreams::filtering_streambuf<boost::iostreams::output>
          out_stream;
      out_stream.push(boost::iostreams::gzip_compressor());
      out_stream.push(out_file);
      std::ostream out(&out_stream);
      out.write((char*)&cfg.N, sizeof(cfg.N));
      CHECK(out.good());
      out.write((char*)&cfg.heldout_ratio, sizeof(cfg.heldout_ratio));
      CHECK(out.good());
      out.write((char*)&num_edges, sizeof(num_edges));
      CHECK(out.good());
      out.write((char *)unique_edges.data(),
                unique_edges.size() * sizeof(mcmc::Edge));
      CHECK(out.good());
      return 0;
    }
  } else {
    std::ifstream in_file(loadFile, std::ifstream::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::input> in_stream;
    in_stream.push(boost::iostreams::gzip_decompressor());
    in_stream.push(in_file);
    std::istream in(&in_stream);
    in.read((char*)&cfg.N, sizeof(cfg.N));
    CHECK(in.good());
    in.read((char*)&cfg.heldout_ratio, sizeof(cfg.heldout_ratio));
    CHECK(in.good());
    size_t num_edges = 0;
    in.read((char*)&num_edges, sizeof(num_edges));
    CHECK(in.good());
    unique_edges.resize(num_edges);
    in.read((char*)unique_edges.data(),
        unique_edges.size() * sizeof(mcmc::Edge));
    CHECK(in.good());
    if (!mcmc::GenerateSetsFromEdges(cfg.N, unique_edges, cfg.heldout_ratio,
                                     &cfg.training_edges, &cfg.heldout_edges,
                                     &cfg.training, &cfg.heldout)) {
      LOG(FATAL) << "Failed to generate training/heldout sets";
    }
  }
  cfg.trainingGraph.reset(new mcmc::Graph(cfg.N, cfg.training_edges));
  cfg.heldoutGraph.reset(new mcmc::Graph(cfg.N, cfg.heldout_edges));

  if (cfg.alpha == 0) cfg.alpha = static_cast<mcmc::Float>(1) / cfg.K;
  cfg.E = unique_edges.size();
  LOG(INFO) << "Loaded file " << (loadDataset ? loadFile : filename)
            << " (training max fan out = " << cfg.trainingGraph->MaxFanOut()
            << ", heldout max fan out = " << cfg.heldoutGraph->MaxFanOut() << ")";
  LOG(INFO) << cfg;
  signal(SIGINT, handler);
  mcmc::Learner learner(cfg, queue);
  LOG(INFO) << "ppx[0] = " << learner.HeldoutPerplexity();
  for (uint64_t i = 0; i < max_iters && !signaled; i += cfg.ppx_interval) {
    uint64_t step = std::min<uint64_t>(max_iters - i, cfg.ppx_interval);
    learner.Run(step);
    if (!signaled) {
      LOG(INFO) << "ppx[" << i + step << "] = " << learner.HeldoutPerplexity();
    }
  }
  LOG_IF(INFO, signaled) << "FORCED TERMINATE";
  learner.PrintStats();
  return 0;
}
