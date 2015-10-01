#include <iostream>
#include <memory>
#include <glog/logging.h>

#include "mcmc/cuckoo.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " [filename]";
  }
  std::unique_ptr<mcmc::CuckooSet> training;
  std::unique_ptr<mcmc::CuckooSet> heldout;
  if (!mcmc::GenerateCuckooSetsFromFile(argv[1], 0.1, &training, &heldout)) {
    LOG(FATAL) << "Failed";
  }
  LOG(INFO) << "DONE";
  return 0;
}

