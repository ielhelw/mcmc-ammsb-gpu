#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "mcmc/learner.h"
#include "mcmc/data.h"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace clcuda = mcmc::clcuda;

struct State {};

clcuda::Device ChooseDevice() {
  // FIXME FIXME
  clcuda::Platform platform((size_t)0);
  return clcuda::Device(platform, 0);
#if 0
  auto devices = compute::system::devices();
  if (devices.size() == 1) {
    return devices[0];
  }
  for (uint32_t i = 0; i < devices.size(); ++i) {
    cout << i << ": " << devices[i].platform().name() << " | "
         << devices[i].name() << endl;
  }
  cout << "Select device: ";
  cout.flush();
  uint32_t choice;
  cin >> choice;
  return devices[choice];
#endif
}

int main(int argc, char **argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  std::string filename;
  mcmc::Config cfg;
  po::options_description options;
  uint32_t max_iters;
  options.add_options()
    ("help,h", "Show usage")
    ("file,f", po::value(&filename)->required(),
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
  ;
  po::variables_map options_vm;
  po::store(po::parse_command_line(argc, argv, options), options_vm);
  if (options_vm.count("help")) {
    std::cout << options << std::endl;
    return 1;
  }
  po::notify(options_vm);
  LOG_IF(FATAL, !fs::exists(filename) || fs::is_directory(filename))
      << "Failed to detect file: " << filename;
  clcuda::Device dev = ChooseDevice();
  clcuda::Context context(dev);
  clcuda::Queue queue(context, dev);
  LOG(INFO) << "OpenCL:" << endl
    << "  Platform: " << dev.Vendor() << endl
    << "  Device: " << dev.Name() << endl
    << "  Device Driver: " << dev.Version();
  std::vector<mcmc::Edge> unique_edges;
  if (!mcmc::GetUniqueEdgesFromFile(filename, &cfg.N, &unique_edges) ||
      !mcmc::GenerateSetsFromEdges(cfg.N, unique_edges, cfg.heldout_ratio,
                                   &cfg.training_edges, &cfg.heldout_edges,
                                   &cfg.training, &cfg.heldout)) {
    LOG(FATAL) << "Failed to generate sets from file " << filename;
  }
  cfg.trainingGraph.reset(new mcmc::Graph(cfg.N, cfg.training_edges));
  cfg.heldoutGraph.reset(new mcmc::Graph(cfg.N, cfg.heldout_edges));
  if (cfg.alpha == 0) cfg.alpha = static_cast<mcmc::Float>(1)/cfg.K;
  cfg.E = unique_edges.size();
  LOG(INFO) << "Loaded file " << filename;
  LOG(INFO) << cfg;
  mcmc::Learner learner(cfg, queue);
  learner.Run(max_iters);
  learner.PrintStats();
  return 0;
}
