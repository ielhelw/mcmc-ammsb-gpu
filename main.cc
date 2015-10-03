#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/compute/system.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "mcmc/data.h"
//#include "mcmc/random.h"

using namespace std;
namespace compute = boost::compute;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

struct Config {
  std::string filename;

  void Setup(po::options_description *desc) {
    desc->add_options()("file,f", po::value<std::string>(&filename)->required(),
                        "Graph data file");
  }
};

ostream &operator<<(ostream &out, const Config &cfg) {
  out << "Config:" << endl << "  Filename: " << cfg.filename;
  return out;
}

struct State {
};

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
  po::options_description options;
  options.add_options()("help,h", "Show usage");
  Config cfg;
  cfg.Setup(&options);
  po::variables_map options_vm;
  po::store(po::parse_command_line(argc, argv, options), options_vm);
  if (options_vm.count("help")) {
    std::cout << options << std::endl;
    return 1;
  }
  po::notify(options_vm);
  LOG(INFO) << cfg;
  LOG_IF(FATAL, !fs::exists(cfg.filename) || fs::is_directory(cfg.filename))
      << "Failed to detect file: " << cfg.filename;

  compute::device dev = ChooseDevice();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  LOG(INFO) << "OpenCL:" << endl << "  Platform: " << dev.platform().name()
            << endl << "  Device: " << dev.name() << endl
            << "  Device Driver: " << dev.driver_version();
  std::unique_ptr<mcmc::Set> training;
  std::unique_ptr<mcmc::Set> heldout;
  std::vector<mcmc::Edge> unique_edges;
  if (!mcmc::GetUniqueEdgesFromFile(cfg.filename, &unique_edges) ||
      !mcmc::GenerateSetsFromEdges(unique_edges, 0.1, &training,
                                           &heldout)) {
    LOG(FATAL) << "Failed to generate sets from file " << cfg.filename;
  }
  return 0;
}
