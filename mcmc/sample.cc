#include "mcmc/sample.h"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <glog/logging.h>
#include <unordered_set>
#include <queue>

#include "mcmc/config.h"

namespace mcmc {

std::string GetNeighborSamplerSource() {
  return R"%%(
    inline uint h1(uint k, uint capacity) {
      return (k ^ 553105253) % capacity;
    }
    inline uint h2(uint k, uint capacity) {
      return 1 + (capacity << 1);
  //    return 1 + (k ^ 961748941);
    }

    void generate_random_int(
        random_seed_t* seed,
        GLOBAL uint* out,
        uint capacity,
        uint max_id,
        uint node) {
      uint r, val;
      do {
        do {
          r = (uint)randint(seed, 0, max_id);
        } while (r == node);
        uint l1 = h1(r, capacity);
        uint l2 = h2(r, capacity);
        for (uint i = 0;; ++i) {
          uint offset = (l1 + i * l2) % capacity;
          val = out[offset];
          if (val == r) break;
          if (val == max_id + 1) {
            out[offset] = r;
            break;
          }
        }
      } while (val == r);
    }

    KERNEL void generate_random_int_kernel(
        uint num_samples, GLOBAL uint* nodes,
        GLOBAL uint* g_out, uint n, uint num, uint capacity,
        GLOBAL uint* g_packed,
        GLOBAL void* vrand) {
      uint gid = GET_GLOBAL_ID();
      uint gsize = GET_GLOBAL_SIZE();
      if (gid < num_samples) {
        random_seed_t seed = ((GLOBAL Random*)vrand)->base_[GET_GLOBAL_ID()];
        for (uint i = gid; i < num_samples; i += gsize) {
          GLOBAL uint* out = g_out + i * capacity;
          GLOBAL uint* packed = g_packed + i * num;
          uint node = nodes[i];
          for (uint j = 0; j < capacity; ++j) {
            out[j] = n;
          }
          for (uint j = 0; j < num; ++j) {
            generate_random_int(&seed, out, capacity, n-1, node);
          }
          uint count = 0;
          for (uint j = 0; j < capacity && count < num; ++j) {
            if (out[j] != n) {
              packed[count] = out[j];
              count++;
            }
          }
        }
        ((GLOBAL Random*)vrand)->base_[GET_GLOBAL_ID()] = seed;
      }
    }
  )%%";
}

NeighborSampler::NeighborSampler(const Config& cfg, clcuda::Queue queue)
    : cfg_(cfg),
      capacity_(2 * cfg.num_node_sample),
      local_(cfg.neighbor_sampler_wg_size),
      queue_(queue),
      hash_(queue_.GetContext(), std::max(2 * cfg.mini_batch_size,
                                          1 + cfg.trainingGraph->MaxFanOut()) *
                                     capacity_),
      data_(queue_.GetContext(), std::max(2 * cfg.mini_batch_size,
                                          1 + cfg.trainingGraph->MaxFanOut()) *
                                     cfg.num_node_sample),
      prog_(queue.GetContext(),
            ::mcmc::random::GetRandomHeader() + GetNeighborSamplerSource()),
      randFactory_(random::OpenClRandomFactory::New(queue_)),
      rand_(randFactory_->CreateRandom(
          2 * cfg.mini_batch_size * cfg.num_node_sample,
          random::random_seed_t{cfg.neighbor_seed[0], cfg.neighbor_seed[1]})) {
  std::vector<std::string> opts = ::mcmc::GetClFlags();
  clcuda::BuildStatus status = prog_.Build(queue.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_.GetBuildInfo(queue.GetDevice());
  kernel_.reset(new clcuda::Kernel(prog_, "generate_random_int_kernel"));
  kernel_->SetArgument(2, hash_);
  kernel_->SetArgument(3, static_cast<uint32_t>(cfg.N));
  kernel_->SetArgument(4, static_cast<uint32_t>(cfg.num_node_sample));
  kernel_->SetArgument(5, static_cast<uint32_t>(capacity_));
  kernel_->SetArgument(6, data_);
  kernel_->SetArgument(7, rand_->Get());
}

void NeighborSampler::operator()(uint32_t num_samples,
                                 clcuda::Buffer<Vertex>* nodes) {
  kernel_->SetArgument(0, num_samples);
  kernel_->SetArgument(1, *nodes);
  clcuda::Event e;
  uint32_t global =
      std::min(num_samples / local_ + (num_samples % local_ ? 1 : 0),
               ::mcmc::GetMaxGroups() / local_);
  kernel_->Launch(queue_, {global * local_}, {local_}, e);
  queue_.Finish();
}

uint32_t NeighborSampler::HashCapacityPerSample() { return capacity_; }

uint32_t NeighborSampler::DataSizePerSample() { return cfg_.num_node_sample; }

Sample::Sample(const Config& cfg, clcuda::Queue q)
    : queue(q.GetContext(), q.GetDevice()),
      dev_edges(q.GetContext(), cfg.mini_batch_size),
      dev_nodes(q.GetContext(), std::max(2 * cfg.mini_batch_size,
                                         1 + cfg.trainingGraph->MaxFanOut())),
      seed(rand()),
      neighbor_sampler(cfg, clcuda::Queue(q.GetContext(), q.GetDevice())) {}

std::istream& operator>>(std::istream& in, SampleStrategy& strategy) {
  std::string token;
  in >> token;
  if (boost::iequals(token, "NodeLink")) {
    strategy = NodeLink;
  } else if (boost::iequals(token, "NodeNonLink")) {
    strategy = NodeNonLink;
  } else if (boost::iequals(token, "Node")) {
    strategy = Node;
  } else if (boost::iequals(token, "BFLink")) {
    strategy = BFLink;
  } else if (boost::iequals(token, "BFNonLink")) {
    strategy = BFNonLink;
  } else if (boost::iequals(token, "BF")) {
    strategy = BF;
  } else {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value,
        "Invalid SampleStrategy");
  }
  return in;
}

std::string to_string(SampleStrategy s) {
  switch (s) {
    case NodeLink:
      return "NodeLink";
    case NodeNonLink:
      return "NodeNonLink";
    case Node:
      return "Node";
    case BFLink:
      return "BFLink";
    case BFNonLink:
      return "BFNonLink";
    case BF:
      return "BF";
    default:
      LOG(FATAL) << "Invalid strategy";
  }
}

Float sampleBreadthFirstNonLink(const Config& cfg, std::vector<Edge>* edges,
                                unsigned int* seed) {
  std::unordered_set<Vertex> Us;
  std::queue<Vertex> q;
  std::unordered_set<Edge> set;
  while (set.size() < cfg.mini_batch_size) {
    if (q.empty()) {
      Vertex u;
      do {
        u = rand_r(seed) % cfg.N;
      } while (Us.find(u) != Us.end());
      q.push(u);
    }
    Vertex u = q.front();
    q.pop();
    if (Us.insert(u).second) {
      auto& neighbors = cfg.trainingGraph->NeighborsOf(u);
      Vertex v;
      for (uint32_t i = 0; i < 32 && set.size() < cfg.mini_batch_size; ++i) {
        do {
          v = rand_r(seed) % cfg.N;
        } while (u == v || std::find(neighbors.begin(), neighbors.end(), v) !=
                               neighbors.end());
        q.push(v);
        set.insert(MakeEdge(std::min(u, v), std::max(u, v)));
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
  return static_cast<Float>((cfg.N * (cfg.N - 1) / 2.0 - cfg.E) /
                            cfg.mini_batch_size);
}

Float sampleBreadthFirstLink(const Config& cfg, std::vector<Edge>* edges,
                             unsigned int* seed) {
  std::unordered_set<Vertex> Us;
  std::queue<Vertex> q;
  std::unordered_set<Edge> set;
  while (set.size() < cfg.mini_batch_size) {
    if (q.empty()) {
      Vertex u;
      do {
        u = rand_r(seed) % cfg.N;
      } while (Us.find(u) != Us.end());
      q.push(u);
    }
    Vertex u = q.front();
    q.pop();
    if (Us.insert(u).second) {
      auto& neighbors = cfg.trainingGraph->NeighborsOf(u);
      for (auto v : neighbors) {
        if (set.size() < cfg.mini_batch_size) {
          q.push(v);
          set.insert(MakeEdge(std::min(u, v), std::max(u, v)));
        } else {
          break;
        }
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
  return static_cast<Float>(cfg.E) / cfg.mini_batch_size;
}

Float sampleBreadthFirst(const Config& cfg, std::vector<Edge>* edges,
                         unsigned int* seed) {
  if (rand_r(seed) % 2) {
    return sampleBreadthFirstLink(cfg, edges, seed);
  } else {
    return sampleBreadthFirstNonLink(cfg, edges, seed);
  }
}

// 1- randomly select u
// 2- select all training edges (u, v)
// 3 repeat until mini-batch is full
Float sampleNodeLink(const Config& cfg, std::vector<Edge>* edges,
                     unsigned int* seed) {
  std::unordered_set<Vertex> Us;
  std::unordered_set<Edge> set;
  while (set.empty()) {
    //  while (set.size() < cfg.mini_batch_size) {
    Vertex u = rand_r(seed) % cfg.N;
    if (Us.insert(u).second) {
      auto& neighbors = cfg.trainingGraph->NeighborsOf(u);
      for (auto& v : neighbors) {
        if (set.size() < cfg.mini_batch_size) {
          Edge e = MakeEdge(std::min(u, v), std::max(u, v));
          set.insert(e);
        } else {
          break;
        }
      }
    }
  }
  edges->insert(edges->begin(), set.begin(), set.end());
  return static_cast<Float>(cfg.N);
}

// 1- randomly select u
// 2- ramdomly select multiple Vs
// 3- make sure (u, v) not in training or heldout
// 4- repeat until mini-batch is full
Float sampleNodeNonLink(const Config& cfg, std::vector<Edge>* edges,
                        unsigned int* seed) {
  std::unordered_set<Edge> set;
  std::unordered_set<Vertex> Vs;
  Vertex u = rand_r(seed) % cfg.N;
  Vertex v;
  Edge e;
  while (set.size() < cfg.mini_batch_size) {
    do {
      v = rand_r(seed) % cfg.N;
      e = MakeEdge(std::min(u, v), std::max(u, v));
    } while (Vs.find(v) != Vs.end() || cfg.heldout->Has(e) ||
             cfg.training->Has(e));
    set.insert(e);
  }
  edges->insert(edges->begin(), set.begin(), set.end());
  // return (cfg.N * cfg.N) / static_cast<Float>(cfg.mini_batch_size);
  return (2 * cfg.E) / static_cast<Float>(cfg.mini_batch_size);
}

Float sampleNode(const Config& cfg, std::vector<Edge>* edges,
                 unsigned int* seed) {
  if (rand_r(seed) % 2) {
    return sampleNodeLink(cfg, edges, seed);
  } else {
    return sampleNodeNonLink(cfg, edges, seed);
  }
}

}  // namespace mcmc
