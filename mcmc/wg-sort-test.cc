#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <regex>
#include <limits>

#include "mcmc/algorithm/sort.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class WgSortTest : public ContextTest {
 protected:
  WgSortTest() : ContextTest(mcmc::algorithm::WorkGroupSort("uint")) {}
};

TEST_F(WgSortTest, Sort) {
  BuildProgram(256);
  clcuda::Kernel kernel(*prog_, "WG_SORT_uint");
  clcuda::Buffer<uint32_t> in(*context_, 256);
  clcuda::Buffer<uint32_t> out(*context_, in.GetSize() / sizeof(uint32_t));
  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> dist(0, 1023);
  auto rand = std::bind(dist, gen);
  std::vector<uint32_t> host(in.GetSize() / sizeof(uint32_t));
  std::generate(host.begin(), host.end(), rand);
  in.Write(*queue_, host.size(), host);
  std::vector<uint32_t> host_sorted(host.begin(), host.end());
  std::sort(host_sorted.begin(), host_sorted.end());
  kernel.SetArgument(0, in);
  kernel.SetArgument(1, out);
  kernel.SetArgument(2, static_cast<uint32_t>(host.size()));
  clcuda::Event e;
  kernel.Launch(*queue_, {host.size()}, {host.size()}, e);
  queue_->Finish();
  out.Read(*queue_, host.size(), host);
  ASSERT_EQ(host_sorted, host);
}

}  // namespace test
}  // namespace mcmc
