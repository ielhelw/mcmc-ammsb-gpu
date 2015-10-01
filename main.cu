#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include "mcmc/cuckoo.h"
#include "mcmc/random.h"

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " [filename]";
  }
  std::unique_ptr<mcmc::CuckooSet> training;
  std::unique_ptr<mcmc::CuckooSet> heldout;
  if (!mcmc::GenerateCuckooSetsFromFile(argv[1], 0.1, &training, &heldout)) {
    LOG(FATAL) << "Failed to generate cuckoo sets from file " << argv[1];
  }
  thrust::device_vector<mcmc::CuckooSet::Element> dev_training_set = training->Serialize();
  thrust::device_vector<mcmc::CuckooSet::Element> dev_heldout_set = heldout->Serialize();
  

  LOG(INFO) << "DONE";
  return 0;
}

