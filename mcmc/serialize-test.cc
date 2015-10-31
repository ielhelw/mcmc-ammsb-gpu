#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <vector>

#include "mcmc/serialize.h"
#include "mcmc/learner.h"
#include "mcmc/test.h"

namespace mcmc {
namespace test {

class SerializeTest : public ContextTest {
 protected:
  SerializeTest() {}
};

TEST_F(SerializeTest, VectorStorage) {
  typedef int T;
  int64_t size = 1024;
  std::vector<T> hbuf(size);
  std::generate(hbuf.begin(), hbuf.end(), rand);
  clcuda::Buffer<T> buf_in(*context_, *queue_, hbuf.begin(), hbuf.end());
  std::ostringstream out;
  ASSERT_TRUE(::mcmc::Serialize(&out, &buf_in, queue_.get()));
  std::istringstream in(out.str());
  clcuda::Buffer<T> buf_out(*context_, size);
  ASSERT_TRUE(::mcmc::Parse(&in, &buf_out, queue_.get()));
  std::vector<T> hbuf_out(size);
  buf_out.Read(*queue_, size, hbuf_out.data());
  ASSERT_EQ(hbuf, hbuf_out);
}

TEST_F(SerializeTest, RowPartitionedMatrix) {
  typedef int T;
  int64_t rows = 211, cols = 1330, rows_in_block = 27;
  auto factory = ::mcmc::RowPartitionedMatrixFactory<T>::New(*queue_);
  auto mat_in = factory->CreateMatrix(rows, cols, rows_in_block);
  std::vector<std::vector<T>> hbufs(mat_in->Blocks().size());
  for (uint32_t i = 0; i < mat_in->Blocks().size(); ++i) {
    hbufs[i].resize(mat_in->Blocks()[i].GetSize() / sizeof(T));
    std::generate(hbufs[i].begin(), hbufs[i].end(), rand);
    mat_in->Blocks()[i].Write(*queue_, hbufs[i].size(), hbufs[i]);
  }
  std::ostringstream out;
  ASSERT_TRUE(::mcmc::Serialize(&out, mat_in, queue_.get()));
  std::istringstream in(out.str());
  // verify
  auto mat_out = factory->CreateMatrix(rows, cols, rows_in_block);
  ASSERT_TRUE(::mcmc::Parse(&in, mat_out, queue_.get()));
  std::vector<std::vector<T>> hbufs_out(mat_out->Blocks().size());
  for (uint32_t i = 0; i < mat_out->Blocks().size(); ++i) {
    hbufs_out[i].resize(mat_out->Blocks()[i].GetSize() / sizeof(T));
    mat_out->Blocks()[i].Read(*queue_, hbufs_out[i].size(), hbufs_out[i]);
    ASSERT_EQ(hbufs[i], hbufs_out[i]);
  }
}

TEST_F(SerializeTest, MultipleMessages) {
  typedef int T;
  int64_t size = 1024;
  uint64_t num_vecs = 10;
  std::vector<std::vector<T>> hbufs(num_vecs);
  std::vector<std::unique_ptr<clcuda::Buffer<T>>> bufs_in(num_vecs);
  for (uint64_t i = 0; i < num_vecs; ++i) {
    hbufs[i].resize(size);
    std::generate(hbufs[i].begin(), hbufs[i].end(), rand);
    bufs_in[i].reset(new clcuda::Buffer<T>(*context_, *queue_, hbufs[i].begin(),
                                           hbufs[i].end()));
  }
  std::ostringstream out;
  for (uint64_t i = 0; i < num_vecs; ++i) {
    ASSERT_TRUE(::mcmc::Serialize(&out, bufs_in[i].get(), queue_.get()));
  }
  std::vector<std::unique_ptr<clcuda::Buffer<T>>> bufs_out(num_vecs);
  std::istringstream in(out.str());
  for (uint64_t i = 0; i < num_vecs; ++i) {
    bufs_out[i].reset(new clcuda::Buffer<T>(*context_, size));
    ASSERT_TRUE(::mcmc::Parse(&in, bufs_out[i].get(), queue_.get()));
  }
  std::vector<std::vector<T>> hbufs_out(num_vecs);
  for (uint64_t i = 0; i < num_vecs; ++i) {
    hbufs_out[i].resize(size);
    bufs_out[i]->Read(*queue_, size, hbufs_out[i].data());
    ASSERT_EQ(hbufs[i], hbufs_out[i]);
  }
}

TEST_F(SerializeTest, EndToEnd) {
  uint32_t iters = 10;
  ::mcmc::Config cfg;
  cfg.N = 1024;
  cfg.heldout_ratio = 0.1;
  cfg.ppx_interval = 2 * iters - 1;
  std::vector<Edge> unique_edges(1024);
  for (auto& e : unique_edges) {
    ::mcmc::Vertex u, v;
    u = rand() % cfg.N;
    do {
      v = rand() % cfg.N;
    } while (u == v);
    e = ::mcmc::MakeEdge(std::min(u, v), std::max(u, v));
  }
  std::sort(unique_edges.begin(), unique_edges.end());
  auto it = std::unique(unique_edges.begin(), unique_edges.end());
  unique_edges.resize(it - unique_edges.begin());
  ASSERT_TRUE(::mcmc::GenerateSetsFromEdges(
      cfg.N, unique_edges, cfg.heldout_ratio, &cfg.training_edges,
      &cfg.heldout_edges, &cfg.training, &cfg.heldout));
  cfg.trainingGraph.reset(new ::mcmc::Graph(cfg.N, cfg.training_edges));
  cfg.heldoutGraph.reset(new ::mcmc::Graph(cfg.N, cfg.heldout_edges));
  if (cfg.alpha == 0) cfg.alpha = static_cast<::mcmc::Float>(1) / cfg.K;
  ASSERT_GT(cfg.training_edges.size(), static_cast<size_t>(0));
  ASSERT_GT(cfg.heldout_edges.size(), static_cast<size_t>(0));
  cfg.E = unique_edges.size();
  LOG(INFO) << cfg;
  std::ostringstream out;
  Float ppx;
  {
    ::mcmc::Learner learner1(cfg, *queue_);
    learner1.Run(iters);
    learner1.Serialize(&out);
    learner1.Run(iters);
    ppx = learner1.HeldoutPerplexity();
  }
  {
    std::istringstream in(out.str());
    ::mcmc::Learner learner2(cfg, *queue_);
    learner2.Parse(&in);
    learner2.Run(iters);
    ASSERT_EQ(ppx, learner2.HeldoutPerplexity());
  }
}

}  // namespace test
}  // namespace mcmc
