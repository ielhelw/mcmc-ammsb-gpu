#include "mcmc/random.h"

#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

using namespace mcmc::random;
namespace compute = boost::compute;

const std::string kSource =
    ::mcmc::random::internal::GetRandomTypes() +
    BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void test(__global void* vrand, __global ulong* ok) {
          __global Random* rand = (__global Random*)vrand;
          if (rand == 0) {
            *ok = 0;
            return;
          }
          if (rand->num_seeds != 10) {
            *ok = 0;
            return;
          }
          for (uint i = 0; i < rand->num_seeds; ++i) {
            if (rand->base_[i].x != 42 + i || rand->base_[i].y != 43 + i) {
              *ok = 0;
              return;
            }
          }
          *ok = 1;
        });

TEST(RandomTest, Check) {
  compute::device dev = compute::system::default_device();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  random_seed_t seed;
  seed[0] = 42;
  seed[1] = 43;
  std::vector<random_seed_t> host(10);
  auto factory = OpenClRandomFactory::New(queue);
  std::unique_ptr<OpenClRandom> random(
      factory->CreateRandom(host.size(), seed));
  compute::copy(random->GetSeeds().begin(), random->GetSeeds().end(),
                host.begin(), queue);
  for (size_t i = 0; i < host.size(); ++i) {
    ASSERT_EQ(seed[0] + i, host[i][0]);
    ASSERT_EQ(seed[1] + i, host[i][1]);
  }

  compute::program prog =
      compute::program::create_with_source(kSource, context);
  try {
    prog.build();
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog.build_log();
  }
  compute::vector<uint64_t> dev_ok(1, static_cast<uint64_t>(0), queue);
  compute::kernel kernel = prog.create_kernel("test");
  kernel.set_arg(0, random->Get());
  kernel.set_arg(1, dev_ok);
  compute::event e = queue.enqueue_task(kernel);
  e.wait();
  std::vector<uint64_t> ok(dev_ok.size(), 0);
  compute::copy(dev_ok.begin(), dev_ok.end(), ok.begin(), queue);
  ASSERT_EQ(1, ok[0]);
}
