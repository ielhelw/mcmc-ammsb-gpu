#ifndef __MCMC_TEST_H__
#define __MCMC_TEST_H__

#include <boost/compute/system.hpp>
#include "mcmc/types.h"

namespace mcmc {
namespace test {

class ContextTest : public ::testing::Test {
 protected:
  ContextTest() {}

  ContextTest(const std::string& source) : source_(source) {}

  void SetUp() override {
    device_ = compute::system::default_device();
    LOG(INFO) << "DEVICE: " << device_.name() << " | "
              << device_.platform().name();
    context_ = compute::context(device_);
    queue_ = compute::command_queue(context_, device_,
                                    compute::command_queue::enable_profiling);
    if (!source_.empty()) {
      prog_ = compute::program::create_with_source(source_, context_);
      try {
        prog_.build();
      } catch (compute::opencl_error& e) {
        LOG(FATAL) << prog_.build_log();
      }
    }
  }

  void TearDown() override {
    prog_ = compute::program();
    device_ = compute::device();
    context_ = compute::context();
    queue_ = compute::command_queue();
  }

  compute::device device_;
  compute::context context_;
  compute::command_queue queue_;
  compute::program prog_;
  std::string source_;
};

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vals) {
  for (size_t i = 0; i < vals.size(); ++i) {
    out << std::setw(12) << std::setfill(' ') << vals[i] << ",";
  }
  out << std::endl;
  return out;
}

}  // namespace test
}  // namespace mcmc

#endif  // __MCMC_TEST_H__
