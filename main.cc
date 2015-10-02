#include <iostream>
#include <memory>
#include <glog/logging.h>
#include <boost/compute/system.hpp>

#include "mcmc/cuckoo.h"
//#include "mcmc/random.h"

using namespace std;
namespace compute = boost::compute;

std::string kSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

    __kernel void testFind(__global void *vset, __global uint64_t *in,
                           uint32_t len, __global uint64_t *out) {
      size_t id = get_global_id(0);
      if (id < len) {
        __global Set *set = (__global Set *)vset;
        out[id] = Set_HasEdge(set, in[id]) ? 1 : 0;
      }
    });

int main(int argc, char **argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    LOG(FATAL) << "Usage: " << argv[0] << "[filename]";
  }

  auto devices = compute::system::devices();
  for (uint32_t i = 0; i < devices.size(); ++i) {
    cout << i << ": " << devices[i].platform().name() << " | "
         << devices[i].name() << endl;
  }
  cout << "Select device: ";
  cout.flush();
  uint32_t choice;
  cin >> choice;

  compute::device dev = devices[choice];
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);

  LOG(INFO) << dev.platform().name();
  LOG(INFO) << dev.name();

  std::unique_ptr<mcmc::CuckooSet> training;
  std::unique_ptr<mcmc::CuckooSet> heldout;
  std::vector<mcmc::Edge> unique_edges;
  if (!mcmc::GetUniqueEdgesFromFile(argv[1], &unique_edges) ||
      !mcmc::GenerateCuckooSetsFromEdges(unique_edges, 0.1, &training,
                                         &heldout)) {
    LOG(FATAL) << "Failed to generate cuckoo sets from file " << argv[1];
  }
  std::vector<mcmc::Edge> training_set = training->Serialize();

  const int NUM = static_cast<int>(ceil(0.8 * unique_edges.size()));
  const int WG_SIZE = 64;
  const int GLOBAL =
      static_cast<int>(std::ceil(static_cast<double>(NUM) / WG_SIZE) * WG_SIZE);
  LOG(INFO) << "NUM=" << NUM << ", GLOBAL=" << GLOBAL;
  LOG_IF(FATAL, GLOBAL < NUM) << "Shape is incorrect";

  compute::vector<mcmc::Edge> dev_input(unique_edges.rbegin(),
                                        unique_edges.rbegin() + NUM, queue);
  compute::vector<mcmc::Edge> dev_output(dev_input.size(), 0, queue);
  compute::program prog =
      compute::program::create_with_source(mcmc::kHeader + kSource, context);
  try {
    prog.build();
  } catch (compute::opencl_error &e) {
    LOG(FATAL) << prog.build_log();
    ;
  }
  auto set_factory = mcmc::OpenClCuckooSetFactory::New(queue);
  std::unique_ptr<mcmc::OpenClCuckooSet> dev_set(
      set_factory->CreateSet(training_set));
  compute::kernel kernel = prog.create_kernel("testFind");
  int arg = 0;
  kernel.set_arg(arg++, dev_set->Get());
  kernel.set_arg(arg++, dev_input);
  kernel.set_arg(arg++, static_cast<uint32_t>(dev_input.size()));
  kernel.set_arg(arg++, dev_output);
  compute::event e = queue.enqueue_1d_range_kernel(kernel, 0, GLOBAL, WG_SIZE);
  LOG(INFO) << "Waiting";
  e.wait();
  LOG(INFO) << "DONE IN " << e.duration<boost::chrono::nanoseconds>().count();
  LOG(INFO) << "Verifying";
  std::vector<mcmc::Edge> ret(dev_output.size());
  compute::copy(dev_output.begin(), dev_output.end(), ret.begin(), queue);
  uint64_t count = 0;
  for (auto v : ret) {
    if (!v) ++count;
  }
  LOG_IF(ERROR, count != 0) << "Failed: missed " << count << " elements";
  return 0;
}
