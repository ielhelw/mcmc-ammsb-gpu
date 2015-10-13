#include "mcmc/cuckoo.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include <glog/logging.h>

namespace mcmc {
namespace cuckoo {

namespace internal {

const std::string& GetSetTypes() {
  static const std::string kClSetTypes = BOOST_COMPUTE_STRINGIZE_SOURCE(

      typedef ulong uint64_t; typedef uint uint32_t;

      typedef struct {
        __global uint64_t* base_;
        uint64_t num_bins_;
      } Set;

      );
  return kClSetTypes;
}

const std::string& GetSetHeader() {
  static const std::string kClSetHeader =
      GetSetTypes() +
      BOOST_COMPUTE_STRINGIZE_SOURCE(

          __constant uint64_t NUM_BUCKETS = 2;
          __constant uint64_t NUM_SLOTS = 4;

          uint64_t hash1(uint64_t num_bins,
                         uint64_t k) { return (1003 * k) % num_bins; }

          uint64_t hash2(uint64_t num_bins,
                         uint64_t k) { return (k ^ 179440147) % num_bins; }

          bool Set_SlotHasEdge_(__global uint64_t * slot, uint64_t k) {
            if (k == slot[0]) return true;
            if (k == slot[1]) return true;
            if (k == slot[2]) return true;
            if (k == slot[3]) return true;
            return false;
          }

          bool Set_HasEdge(__global Set * set, uint64_t k) {
            uint64_t h1 = hash1(set->num_bins_, k);
            if (Set_SlotHasEdge_(set->base_ + 0 * set->num_bins_ * NUM_SLOTS +
                                     h1 * NUM_SLOTS,
                                 k))
              return true;
            uint64_t h2 = hash2(set->num_bins_, k);
            if (Set_SlotHasEdge_(set->base_ + 1 * set->num_bins_ * NUM_SLOTS +
                                     h2 * NUM_SLOTS,
                                 k))
              return true;
            return false;
          }

          );
  return kClSetHeader;
}

const std::string& GetSetSource() {
  static const std::string kClSetSource =
      GetSetTypes() +
      BOOST_COMPUTE_STRINGIZE_SOURCE(

          __kernel void SizeOfSet(__global uint64_t *
                                  size) { *size = sizeof(Set); }

          __kernel void SetInit(__global void* vset, __global uint64_t* data,
                                uint64_t num_bins) {
            __global Set* set = (__global Set*)vset;
            set->base_ = data;
            set->num_bins_ = num_bins;
          }

          );
  return kClSetSource;
}

}  // namespace internal

const Edge Set::KEY_INVALID = std::numeric_limits<Edge>::max();

Set::Set(size_t n)
    : count_(0),
      N_(static_cast<size_t>(
          1 + std::ceil((1.27 * n) / (NUM_BUCKETS * NUM_SLOTS)))),
      seed_(42),
      displacements_max_(n / 2 + 1) {
  for (auto& bucket : buckets_) {
    bucket.resize(N_);
    for (auto& slot : bucket) {
      for (auto& val : slot) {
        val = KEY_INVALID;
      }
    }
  }
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
      return (1003 * k) % N_;
    case 1:
      return (k ^ 179440147) % N_;
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
                     compute::kernel* init, compute::command_queue* queue,
                     uint64_t sizeOfSet, const std::vector<uint64_t>& data)
    : factory_(factory),
      queue_(*queue),
      data_(data, queue_),
      num_bins_(data_.size() / (Set::NUM_BUCKETS * Set::NUM_SLOTS)),
      buf_(compute::buffer(queue_.get_context(), sizeOfSet,
                           compute::memory_object::read_write)) {
  init->set_arg(0, buf_);
  init->set_arg(1, data_);
  init->set_arg(2, num_bins_);
  auto e = queue_.enqueue_task(*init);
  e.wait();
}

const std::string& OpenClSetFactory::GetHeader() {
  return internal::GetSetHeader();
}

std::shared_ptr<OpenClSetFactory> OpenClSetFactory::New(
    compute::command_queue queue) {
  return std::shared_ptr<OpenClSetFactory>(new OpenClSetFactory(queue));
}

OpenClSet* OpenClSetFactory::CreateSet(const std::vector<uint64_t>& data) {
  return new OpenClSet(shared_from_this(), &init_kernel_, &queue_, sizeOfSet_,
                       data);
}

OpenClSetFactory::OpenClSetFactory(compute::command_queue queue)
    : queue_(queue) {
  prog_ = compute::program::create_with_source(internal::GetSetSource(),
                                               queue_.get_context());
  try {
    prog_.build();
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog_.build_log();
    ;
  }
  init_kernel_ = prog_.create_kernel("SetInit");
  compute::kernel sizeOf_kernel = prog_.create_kernel("SizeOfSet");
  compute::vector<uint64_t> size(1, (uint64_t)0, queue_);
  sizeOf_kernel.set_arg(0, size);
  auto e = queue_.enqueue_task(sizeOf_kernel);
  e.wait();
  compute::copy(size.begin(), size.end(), &sizeOfSet_, queue_);
}

}  // namespace cuckoo
}  // namespace mcmc
