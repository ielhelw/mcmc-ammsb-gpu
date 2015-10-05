#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/compute/system.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "mcmc/learner.h"

using namespace std;
namespace compute = boost::compute;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

struct State {};

compute::device ChooseDevice() {
  auto devices = compute::system::devices();
  for (uint32_t i = 0; i < devices.size(); ++i) {
    cout << i << ": " << devices[i].platform().name() << " | "
         << devices[i].name() << endl;
  }
  cout << "Select device: ";
  cout.flush();
  uint32_t choice;
  cin >> choice;
  return devices[choice];
}

int main(int argc, char **argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  std::string filename;
  mcmc::Config cfg;
  po::options_description options;
  options.add_options()
    ("help,h", "Show usage")
    ("file,f", po::value(&filename)->required(),
      "Graph data file")
    ("heldout-ratio,r", po::value(&cfg.heldout_ratio)->default_value(0.01))
    ("alpha", po::value(&cfg.alpha)->default_value(0))
    ("a,a", po::value(&cfg.a)->default_value(0.0315))
    ("b,b", po::value(&cfg.b)->default_value(1024))
    ("c,c", po::value(&cfg.c)->default_value(0.5))
    ("epsilon,e", po::value(&cfg.epsilon)->default_value(1e-7))
    ("k,k", po::value(&cfg.K)->default_value(32))
    ("mini_batch,m", po::value(&cfg.mini_batch_size)->default_value(32))
    ("neighbors,n", po::value(&cfg.num_node_sample)->default_value(32))
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

  compute::device dev = ChooseDevice();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  LOG(INFO) << "OpenCL:" << endl
    << "  Platform: " << dev.platform().name() << endl
    << "  Device: " << dev.name() << endl
    << "  Device Driver: " << dev.driver_version();
  std::vector<mcmc::Edge> unique_edges;
  if (!mcmc::GetUniqueEdgesFromFile(filename, &cfg.N, &unique_edges) ||
      !mcmc::GenerateSetsFromEdges(unique_edges, cfg.heldout_ratio, &cfg.training,
                                   &cfg.heldout)) {
    LOG(FATAL) << "Failed to generate sets from file " << filename;
  }
  if (cfg.alpha == 0) cfg.alpha = static_cast<mcmc::Float>(1)/cfg.K;
  cfg.E = unique_edges.size();
  LOG(INFO) << cfg;
  LOG(INFO) << "Loaded file " << filename;
  return 0;
}
