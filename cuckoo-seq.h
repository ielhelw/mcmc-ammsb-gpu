#ifndef __CUCKOO_H__
#define __CUCKOO_H__

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace mcmc {

class CuckooSet {
 public:
  static const size_t NUM_BUCKETS = 2;
  
  typedef uint32_t Element;
  typedef std::vector<Element> Slot;
  typedef std::vector<Slot> Bucket;
  typedef std::array<Bucket, NUM_BUCKETS> Buckets;
  
  static const Element KEY_INVALID = std::numeric_limits<Element>::max();

  static const size_t MAX_DISPLACEMENTS = 1024;

  CuckooSet(size_t n, size_t num_slots = 4)
    : N_(static_cast<size_t>(std::ceil((1.5 * n ) / (NUM_BUCKETS * num_slots))))
#ifndef NDEBUG
      , displacements_max_(0), displacements_count_(0), displacements_avg_(0)
#endif
  {
    for (auto& bucket : buckets_) {
      bucket.resize(N_);
      for (auto& slot : bucket) {
        slot.resize(num_slots);
        for (auto& val : slot) {
          val = KEY_INVALID;
        }
      } 
    }
  }
 
  bool IsSlotNotFullAndKeyNotInIt(Element k, const Slot& slot) {
    bool slotFull = true;
    for (auto v : slot) {
      if (v == KEY_INVALID) slotFull = false;
      if (v == k) return false;
    }
    return !slotFull;
  }

  bool Insert(Element k) {
    size_t displacements = 0;
    do {
      for (size_t i = 0; i < buckets_.size(); ++i) {
        size_t h = Hash(k, i);
        if (IsSlotNotFullAndKeyNotInIt(k, buckets_[i][h])) {
//      if (!IsSlotFull(buckets_[i][h]) && !IsKeyInSlot(k, buckets_[i][h])) {
          Element ret = InsertKeyInSlot(k, &buckets_[i][h]);
          assert(ret == KEY_INVALID);
#ifndef NDEBUG
          displacements_max_ = std::max(displacements, displacements_max_);
          ++displacements_count_;
          displacements_avg_ = static_cast<double>(displacements_avg_ * (displacements_count_ - 1) + displacements) / displacements_count_;
#endif
          return true;
        }
      }
      size_t bidx = rand()%NUM_BUCKETS;
      size_t h = Hash(k, bidx);
      k = InsertKeyInSlot(k, &buckets_[bidx][h]);
      assert(k != KEY_INVALID);
    } while (++displacements < 1024);
    return false;
  }

  bool Has(Element k) {
    size_t bidx = 0;
    for (; bidx < NUM_BUCKETS; ++bidx) {
      size_t idx = Hash(k, bidx);
      if (IsKeyInSlot(k, buckets_[bidx][idx])) return true;
    }
    return false;
  }

#ifndef NDEBUG
  size_t displacements_max() const { return displacements_max_; }
  size_t displacements_count() const { return displacements_count_; }
  double displacements_avg() const { return displacements_avg_; }
#endif
 
 private:
  bool IsSlotFull(const Slot& slot) {
    for (auto v : slot) {
      if (v == KEY_INVALID) return false;
    }
    return true;
  }

  bool IsKeyInSlot(Element k, const Slot& slot) {
    for (auto v : slot) {
      if (v == k) return true;
    }
    return false;
  }
  
  Element InsertKeyInSlot(Element k, Slot* slot) {
    for (auto& v : *slot) {
      if (v == KEY_INVALID) {
        v = k;
        return KEY_INVALID;
      }
    }
    size_t alt_idx = rand()%slot->size();
    Element alt = (*slot)[alt_idx];
    assert(alt != KEY_INVALID);
    (*slot)[alt_idx] = k;
    return alt;
  }

  size_t Hash(Element k, size_t bidx) {
    assert(0 <= bidx && bidx < NUM_BUCKETS);
    switch(bidx) {
      case 0: return k % N_;
      case 1: return 1 + 2 * ((k * 179440147) % (N_ / 2));
    }
  }

  size_t N_;
  Buckets buckets_;
#ifndef NDEBUG
  size_t displacements_max_;
  size_t displacements_count_;
  double displacements_avg_;
#endif
};

} // namespace mcmc

#endif // __CUCKOO_H__
