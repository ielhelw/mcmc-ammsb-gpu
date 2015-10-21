#ifndef __MCMC_TEST_H__
#define __MCMC_TEST_H__

#include "mcmc/types.h"
#include <boost/algorithm/string.hpp>

namespace mcmc {
namespace test {

class ContextTest : public ::testing::Test {
 protected:
  ContextTest() {}

  ContextTest(const std::string& source) : source_(source) {}

  void BuildProgram(uint32_t wg) {
    prog_.reset(new clcuda::Program(*context_, source_));
    std::vector<std::string> opts = ::mcmc::GetClFlags(wg);
    clcuda::BuildStatus status = prog_->Build(*device_, opts);
    if (status != clcuda::BuildStatus::kSuccess) {
      std::vector<std::string> lines;
      boost::split(lines, source_, boost::is_any_of("\n"));
      for (uint32_t i = 0; i < lines.size(); ++i) {
        LOG(ERROR) << std::setw(4) << i << ": " << lines[i];
      }
      LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
          << prog_->GetBuildInfo(*device_);
    }
  }

  void SetUp() override {
    clcuda::Platform platform((size_t)0);
    device_.reset(new clcuda::Device(platform, 0));
    LOG(INFO) << "DEVICE: " << device_->Name() << " | " << device_->Vendor();
    context_.reset(new clcuda::Context(*device_));
    queue_.reset(new clcuda::Queue(*context_, *device_));
  }

  void TearDown() override {
    queue_->Finish();
    queue_.reset();
    prog_.reset();
    context_.reset();
    device_.reset();
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
