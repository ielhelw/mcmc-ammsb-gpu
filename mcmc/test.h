#ifndef __MCMC_TEST_H__
#define __MCMC_TEST_H__

#include "mcmc/types.h"

namespace mcmc {
namespace test {

class ContextTest : public ::testing::Test {
 protected:
  ContextTest() {}

  ContextTest(const std::string& source) : source_(source) {}

  void SetUp() override {
    clcuda::Platform platform((size_t)0);
    device_.reset(new clcuda::Device(platform, 0));
    LOG(INFO) << "DEVICE: " << device_->Name() << " | " << device_->Vendor();
    context_.reset(new clcuda::Context(*device_));
    queue_.reset(new clcuda::Queue(*context_, *device_));
    if (!source_.empty()) {
      prog_.reset(new clcuda::Program(*context_, source_));
      std::vector<std::string> opts;
      clcuda::BuildStatus status = prog_->Build(*device_, opts);
      LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
          << prog_->GetBuildInfo(*device_);
    }
  }

  void TearDown() override {
    prog_.reset();
    device_.reset();
    context_.reset();
    queue_.reset();
  }

  std::unique_ptr<clcuda::Device> device_;
  std::unique_ptr<clcuda::Context> context_;
  std::unique_ptr<clcuda::Queue> queue_;
  std::unique_ptr<clcuda::Program> prog_;
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
