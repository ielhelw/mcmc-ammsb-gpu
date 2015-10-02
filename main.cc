#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/compute/system.hpp>
#include <boost/compute/function.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/cuckoo.h"
//#include "mcmc/random.h"

using namespace std;
namespace compute = boost::compute;

struct SetProperties {
  uint64_t num_bins_per_bucket;
  uint64_t num_slots_per_bin;
  uint64_t num_buckets;
};

BOOST_COMPUTE_ADAPT_STRUCT(SetProperties, SetProperties, (num_bins_per_bucket, num_slots_per_bin, num_buckets))

const std::string source = std::string("#define uint64_t ulong \n #define uint32_t uint \n")
+ compute::type_definition<SetProperties>()
+ "\n"
+ BOOST_COMPUTE_STRINGIZE_SOURCE(

uint64_t hash(uint64_t bidx, uint64_t bins_per_bucket, uint64_t k) {
  switch(bidx) {
    case 0: return (1003 * k) % bins_per_bucket;
    case 1: return (k ^ 179440147) % bins_per_bucket;
  }
  printf("HASH IS BAD\n");
  return 0;
}

bool Set_SlotHasEdge_(__global uint64_t* slot, int slots_per_bin, uint64_t k) {
  for (int i = 0; i < slots_per_bin; ++i) {
    if (k == slot[i]) return true;
  }
  return false;
}

bool Set_BucketHasEdge_(__global uint64_t* set, SetProperties props, uint64_t bidx, uint64_t k) {
  uint64_t h = hash(bidx, props.num_bins_per_bucket, k);
  return Set_SlotHasEdge_(
    set
      + bidx * props.num_bins_per_bucket * props.num_slots_per_bin
      + h * props.num_slots_per_bin,
    props.num_slots_per_bin, k);
}

bool Set_HasEdge(__global uint64_t* set, SetProperties props, uint64_t k) {
  for (uint64_t i = 0; i < props.num_buckets; ++i) {
    if (Set_BucketHasEdge_(set, props, i, k)) return true;
  }
  return false;
}

__kernel void testFind(__global uint64_t* set, __global SetProperties* props, __global uint64_t* in, uint64_t len, __global uint64_t* out) {
  size_t id = get_global_id(0);
  if (id < len) {
    out[id] = Set_HasEdge(set, *props, in[id]) ? 1 : 0;
  }
}

);

int main(int argc, char** argv) {
  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << " [filename]";
  }

  compute::device dev = compute::system::default_device();
  compute::context context(dev);
  compute::command_queue queue(context, dev, compute::command_queue::enable_profiling);

  std::unique_ptr<mcmc::CuckooSet> training;
  std::unique_ptr<mcmc::CuckooSet> heldout;
  std::vector<mcmc::Edge> unique_edges;
  if (!mcmc::GetUniqueEdgesFromFile(argv[1], &unique_edges)
      || !mcmc::GenerateCuckooSetsFromEdges(unique_edges, 0.1, &training, &heldout)) {
    LOG(FATAL) << "Failed to generate cuckoo sets from file " << argv[1];
  }
  std::vector<mcmc::Edge> training_set = training->Serialize();
  std::vector<mcmc::Edge> heldout_set = heldout->Serialize();
  
  compute::vector<mcmc::Edge> dev_training_set(training_set);
  compute::vector<mcmc::Edge> dev_heldout_set(heldout_set);

  const int NUM = 1e6;
  const int WG_SIZE = 32;

  compute::vector<mcmc::Edge> dev_input(unique_edges.rbegin(), unique_edges.rbegin() + NUM);
  compute::vector<mcmc::Edge> dev_output(dev_input.size(), 0);
  SetProperties props = {
    training->BinsPerBucket(),
    training->SlotsPerBin(),
    mcmc::CuckooSet::NUM_BUCKETS
  };
  compute::buffer dev_props(context, sizeof(props),
      compute::memory_object::read_write | compute::memory_object::copy_host_ptr,
      &props);

  compute::program prog = compute::program::create_with_source(source, context);
  try {
    prog.build();
  } catch(compute::opencl_error &e){
    LOG(ERROR) << prog.build_log();;
  }
  compute::kernel kernel = prog.create_kernel("testFind");
  int arg = 0;
  kernel.set_arg(arg++, dev_training_set);
  kernel.set_arg(arg++, dev_props.get());
  kernel.set_arg(arg++, dev_input);
  kernel.set_arg(arg++, dev_input.size());
  kernel.set_arg(arg++, dev_output);
  compute::event e = queue.enqueue_1d_range_kernel(kernel, 0, (NUM/WG_SIZE)*WG_SIZE, WG_SIZE);
  e.wait();

  std::vector<uint64_t> ret(dev_output.size());
  compute::copy(dev_output.begin(), dev_output.end(), ret.begin());
  for (auto v : ret) {
    if (!v) LOG(INFO) << "missing: " << v;
  }
  LOG(INFO) << "DONE IN " << e.duration<boost::chrono::nanoseconds>().count();
  return 0;
}

