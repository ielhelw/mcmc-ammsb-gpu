#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>
#include <map>

#include "mcmc/test.h"
#include "mcmc/random.h"
#include "mcmc/config.h"
#include "mcmc/sample.h"

namespace mcmc {
namespace test {

class WgBTest : public ContextTest {
 protected:
  WgBTest()
      : ContextTest(::mcmc::random::GetRandomHeader() +
                    ::mcmc::GetNeighborSamplerSource()) {}
};

TEST_F(WgBTest, AA) {
  ::mcmc::Config cfg;
  cfg.N = 12000;
  cfg.num_node_sample = 20;
  cfg.mini_batch_size = 32*1024;
  cfg.trainingGraph.reset(new ::mcmc::Graph(cfg.N, {}));
  ::mcmc::NeighborSampler sampler(cfg, *queue_);
  std::vector<Vertex> mini_batch(cfg.mini_batch_size);
  std::generate(mini_batch.begin(), mini_batch.end(),
                [&cfg]() { return rand() % cfg.N; });
  clcuda::Buffer<Vertex> dev_mini_batch(*context_, *queue_, mini_batch.begin(),
                                        mini_batch.end());
  uint64_t nanos = 0;
  uint32_t num_tries = 10;
  for (uint32_t i = 0; i < num_tries; ++i) {
    uint32_t num_samples = 2*cfg.mini_batch_size;
    auto t1 = std::chrono::high_resolution_clock::now();
    sampler(num_samples, &dev_mini_batch);
    auto t2 = std::chrono::high_resolution_clock::now();
    nanos +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    for (uint32_t j = 0; j < num_samples; ++j) {
      std::vector<uint32_t> hvals(sampler.HashCapacityPerSample());
      sampler.GetHash().Read(*queue_, hvals.size(), hvals, j * hvals.size());
      std::vector<uint32_t> hdata(sampler.DataSizePerSample());
      sampler.GetData().Read(*queue_, hdata.size(), hdata, j * hdata.size());
      std::map<uint32_t, uint32_t> m;
      uint32_t IDX = 0;
      for (auto i : hvals) {
        if (i != cfg.N) {
          ASSERT_EQ(i, hdata[IDX++]);
        }
        if (m.find(i) != m.end()) {
          m[i]++;
        } else {
          m[i] = 1;
        }
      }
      for (auto i : m) {
        if (i.second > 1) {
          ASSERT_EQ(cfg.N, i.first);
          ASSERT_EQ(
              sampler.HashCapacityPerSample() - sampler.DataSizePerSample(),
              i.second);
        }
      }
    }
  }
  LOG(INFO) << "AVG SECONDS = " << (static_cast<double>(nanos) / num_tries) /
                                       1e9;
}

}  // namespace test
}  // namespace mcmc
