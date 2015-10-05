#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <regex>
#include <limits>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sort.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class WgSortTest : public ContextTest {
 protected:
  WgSortTest() : ContextTest(mcmc::algorithm::WorkGroupSort("uint")) {}
};

TEST_F(WgSortTest, Sort) {
  compute::kernel kernel = prog_.create_kernel("WG_SORT_uint");
  compute::vector<compute::uint_> in(256, context_);
  compute::vector<compute::uint_> out(in.size(), context_);
  compute::mersenne_twister_engine<compute::uint_> rand(queue_);
  rand.generate(in.begin(), in.end(), queue_);
  std::vector<compute::uint_> host(in.size());
  compute::copy(in.begin(), in.end(), host.begin(), queue_);
  std::vector<compute::uint_> host_sorted(host.begin(), host.end());
  std::sort(host_sorted.begin(), host_sorted.end());
  kernel.set_arg(0, in);
  kernel.set_arg(1, out);
  kernel.set_arg(2, static_cast<compute::uint_>(in.size()));
  kernel.set_arg(3, in.size() * sizeof(compute::uint_), nullptr);
  compute::event e =
      queue_.enqueue_1d_range_kernel(kernel, 0, in.size(), in.size());
  e.wait();
  compute::copy(out.begin(), out.end(), host.begin(), queue_);
  EXPECT_EQ(host_sorted, host);
}

}  // namespace test
}  // namespace mcmc
