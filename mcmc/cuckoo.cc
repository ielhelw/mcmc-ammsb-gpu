#include "mcmc/cuckoo.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <glog/logging.h>

namespace mcmc {
namespace cuckoo {

namespace internal {

const std::string GetSetTypes() {
  static const std::string kClSetTypes = R"%%(

      typedef struct {
        GLOBAL uint64_t* base_;
        uint64_t num_bins_;
        uint32_t primes_idx_;
      } Set;

      )%%";
  return ::mcmc::GetClTypes() + kClSetTypes;
}

const std::string& GetSetHeader() {
  static const std::string kClSetHeader = GetSetTypes() + R"%%(

          CONSTANT uint64_t NUM_BUCKETS = 2;
          CONSTANT uint64_t NUM_SLOTS = 4;
          CONSTANT uint64_t SET_PRIMES[] = {
            15485807, 920429591,
            379906717, 740320571,
            256204747, 379927517,
            13, 17
          };

          uint64_t hash1(uint64_t num_bins,
                         uint64_t k, uint32_t pidx) { return (SET_PRIMES[2*pidx] * k) % num_bins; }

          uint64_t hash2(uint64_t num_bins,
                         uint64_t k, uint32_t pidx) { return (k ^ SET_PRIMES[2*pidx+1]) % num_bins; }

          bool Set_SlotHasEdge_(GLOBAL uint64_t * slot, uint64_t k) {
            if (k == slot[0]) return true;
            if (k == slot[1]) return true;
            if (k == slot[2]) return true;
            if (k == slot[3]) return true;
            return false;
          }

          bool Set_HasEdge(GLOBAL Set * set, uint64_t k) {
            uint64_t h1 = hash1(set->num_bins_, k, set->primes_idx_);
            if (Set_SlotHasEdge_(set->base_ + 0 * set->num_bins_ * NUM_SLOTS +
                                     h1 * NUM_SLOTS,
                                 k))
              return true;
            uint64_t h2 = hash2(set->num_bins_, k, set->primes_idx_);
            if (Set_SlotHasEdge_(set->base_ + 1 * set->num_bins_ * NUM_SLOTS +
                                     h2 * NUM_SLOTS,
                                 k))
              return true;
            return false;
          }

          )%%";
  return kClSetHeader;
}

const std::string& GetSetSource() {
  static const std::string kClSetSource = GetSetTypes() + R"%%(

          KERNEL void SizeOfSet(GLOBAL uint64_t *
                                  size) { *size = sizeof(Set); }

          KERNEL void SetInit(GLOBAL void* vset, GLOBAL uint64_t* data,
                                uint64_t num_bins, uint32_t primes_idx) {
            GLOBAL Set* set = (GLOBAL Set*)vset;
            set->base_ = data;
            set->num_bins_ = num_bins;
            set->primes_idx_ = primes_idx;
          }

          )%%";
  return kClSetSource;
}

}  // namespace internal

const Edge Set::KEY_INVALID = std::numeric_limits<Edge>::max();
const std::vector<std::pair<uint64_t, uint64_t>> Set::PRIMES = {
    std::make_pair<uint64_t, uint64_t>(15485807, 920429591),
    std::make_pair<uint64_t, uint64_t>(379906717, 740320571),
    std::make_pair<uint64_t, uint64_t>(256204747, 379927517),
    std::make_pair<uint64_t, uint64_t>(13, 17)};

Set::Set(size_t n)
    : count_(0),
      N_(static_cast<size_t>(
          1 + std::ceil((1.15 * n) / (NUM_BUCKETS * NUM_SLOTS)))),
      seed_(42),
      displacements_max_(n / 2 + 1),
      primeIdx_(0) {}

void Set::Reset() {
  for (auto& bucket : buckets_) {
    bucket.resize(N_);
    for (auto& slot : bucket) {
      for (auto& val : slot) {
        val = KEY_INVALID;
      }
    }
  }
}

bool Set::SetContents(std::vector<Edge>::const_iterator start,
                      std::vector<Edge>::const_iterator end) {
  for (primeIdx_ = 0; primeIdx_ < PRIMES.size(); ++primeIdx_) {
    Reset();
    bool ok = true;
    for (auto it = start; ok && it != end; ++it) {
      ok = Insert(*it);
    }
    if (ok) return ok;
    LOG(ERROR) << "Attempt(" << primeIdx_ << ") to create Set failed";
  }
  return false;
}

bool Set::IsSlotNotFullAndKeyNotInIt(Edge k, const Slot& slot) const {
  bool slotFull = true;
  for (auto& v : slot) {
    if (v == KEY_INVALID) slotFull = false;
    if (v == k) return false;
  }
  return !slotFull;
}

bool Set::Insert(Edge k) {
  size_t displacements = 0;
  do {
    for (size_t i = 0; i < buckets_.size(); ++i) {
      size_t h = Hash(k, i);
      if (IsSlotNotFullAndKeyNotInIt(k, buckets_[i][h])) {
#ifndef NDEBUG
        Edge ret =
#endif
            InsertKeyInSlot(k, &buckets_[i][h]);
        assert(ret == KEY_INVALID);
        ++count_;
        return true;
      }
    }
    size_t bidx = rand_r(&seed_) % NUM_BUCKETS;
    size_t h = Hash(k, bidx);
    k = InsertKeyInSlot(k, &buckets_[bidx][h]);
    assert(k != KEY_INVALID);
  } while (++displacements < displacements_max_);
  return false;
}

bool Set::Has(Edge k) const {
  size_t bidx = 0;
  for (; bidx < NUM_BUCKETS; ++bidx) {
    size_t idx = Hash(k, bidx);
    if (IsKeyInSlot(k, buckets_[bidx][idx])) return true;
  }
  return false;
}

bool Set::IsSlotFull(const Slot& slot) const {
  for (auto& v : slot) {
    if (v == KEY_INVALID) return false;
  }
  return true;
}

bool Set::IsKeyInSlot(Edge k, const Slot& slot) const {
  for (auto& v : slot) {
    if (v == k) return true;
  }
  return false;
}

Edge Set::InsertKeyInSlot(Edge k, Slot* slot) {
  for (auto& v : *slot) {
    if (v == KEY_INVALID) {
      v = k;
      return KEY_INVALID;
    }
  }
  size_t alt_idx = rand_r(&seed_) % slot->size();
  Edge alt = (*slot)[alt_idx];
  (*slot)[alt_idx] = k;
  return alt;
}

size_t Set::Hash(Edge k, size_t bidx) const {
  assert(bidx < NUM_BUCKETS);
  switch (bidx) {
    case 0:
      return (PRIMES[primeIdx_].first * k) % N_;
    case 1:
      return (k ^ PRIMES[primeIdx_].second) % N_;
    default:
      abort();
  }
}

std::vector<Edge> Set::Serialize() const {
  std::vector<Edge> vals(NUM_BUCKETS * BinsPerBucket() * NUM_SLOTS);
  auto it = vals.begin();
  for (auto& bucket : buckets_) {
    for (auto& slot : bucket) {
      it = std::copy(slot.begin(), slot.end(), it);
    }
  }
  return vals;
}

OpenClSet::OpenClSet(std::shared_ptr<OpenClSetFactory> factory,
                     clcuda::Kernel* init, clcuda::Queue* queue,
                     uint64_t sizeOfSet, const Set& set)
    : factory_(factory),
      queue_(*queue),
      data_(queue_.GetContext(), set.Capacity()),
      num_bins_(set.BinsPerBucket()),
      buf_(queue_.GetContext(), sizeOfSet) {
  auto data = set.Serialize();
  data_.Write(queue_, data.size(), data);
  init->SetArgument(0, buf_);
  init->SetArgument(1, data_);
  init->SetArgument(2, num_bins_);
  init->SetArgument(3, set.PrimeIdx());
  clcuda::Event e;
  init->Launch(queue_, {1}, {1}, e);
  queue_.Finish();
}

const std::string& OpenClSetFactory::GetHeader() {
  return internal::GetSetHeader();
}

std::shared_ptr<OpenClSetFactory> OpenClSetFactory::New(clcuda::Queue queue) {
  return std::shared_ptr<OpenClSetFactory>(new OpenClSetFactory(queue));
}

OpenClSet* OpenClSetFactory::CreateSet(const Set& set) {
  return new OpenClSet(shared_from_this(), init_kernel_.get(), &queue_,
                       sizeOfSet_, set);
}

OpenClSetFactory::OpenClSetFactory(clcuda::Queue queue)
    : queue_(queue), prog_(queue_.GetContext(), internal::GetSetSource()) {
  std::vector<std::string> opts = ::mcmc::GetClFlags();
  clcuda::BuildStatus status = prog_.Build(queue.GetDevice(), opts);
  LOG_IF(FATAL, status != clcuda::BuildStatus::kSuccess)
      << prog_.GetBuildInfo(queue.GetDevice());
  init_kernel_.reset(new clcuda::Kernel(prog_, "SetInit"));
  clcuda::Kernel sizeOf_kernel(prog_, "SizeOfSet");
  clcuda::Buffer<uint64_t> size(queue_.GetContext(), 1);
  sizeOf_kernel.SetArgument(0, size);
  clcuda::Event e;
  sizeOf_kernel.Launch(queue_, {1}, {1}, e);
  queue.Finish();
  size.Read(queue_, 1, &sizeOfSet_);
}

}  // namespace cuckoo
}  // namespace mcmc
