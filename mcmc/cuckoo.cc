#include "mcmc/cuckoo.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

#include <glog/logging.h>

namespace mcmc {

const CuckooSet::Element CuckooSet::KEY_INVALID =
    std::numeric_limits<CuckooSet::Element>::max();

CuckooSet::CuckooSet(size_t n, size_t num_slots)
    : N_(static_cast<size_t>(1 +
                             std::ceil((1.2 * n) / (NUM_BUCKETS * num_slots)))),
      num_slots_per_bin_(num_slots),
      seed_(42),
      displacements_max_(n / 2 + 1) {
  for (auto& bucket : buckets_) {
    bucket.resize(N_);
    for (auto& slot : bucket) {
      slot = Slot(num_slots);
      for (auto& val : slot) {
        val = KEY_INVALID;
      }
    }
  }
}

bool CuckooSet::IsSlotNotFullAndKeyNotInIt(CuckooSet::Element k,
                                           const Slot& slot) const {
  bool slotFull = true;
  for (auto& v : slot) {
    if (v == KEY_INVALID) slotFull = false;
    if (v == k) return false;
  }
  return !slotFull;
}

bool CuckooSet::Insert(CuckooSet::Element k) {
  size_t displacements = 0;
  do {
    for (size_t i = 0; i < buckets_.size(); ++i) {
      size_t h = Hash(k, i);
      if (IsSlotNotFullAndKeyNotInIt(k, buckets_[i][h])) {
#ifndef NDEBUG
        Element ret =
#endif
            InsertKeyInSlot(k, &buckets_[i][h]);
        assert(ret == KEY_INVALID);
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

bool CuckooSet::Has(CuckooSet::Element k) const {
  size_t bidx = 0;
  for (; bidx < NUM_BUCKETS; ++bidx) {
    size_t idx = Hash(k, bidx);
    if (IsKeyInSlot(k, buckets_[bidx][idx])) return true;
  }
  return false;
}

bool CuckooSet::IsSlotFull(const Slot& slot) const {
  for (auto& v : slot) {
    if (v == KEY_INVALID) return false;
  }
  return true;
}

bool CuckooSet::IsKeyInSlot(CuckooSet::Element k, const Slot& slot) const {
  for (auto& v : slot) {
    if (v == k) return true;
  }
  return false;
}

CuckooSet::Element CuckooSet::InsertKeyInSlot(CuckooSet::Element k,
                                              Slot* slot) {
  for (auto& v : *slot) {
    if (v == KEY_INVALID) {
      v = k;
      return KEY_INVALID;
    }
  }
  size_t alt_idx = rand_r(&seed_) % slot->size();
  Element alt = (*slot)[alt_idx];
  (*slot)[alt_idx] = k;
  return alt;
}

size_t CuckooSet::Hash(CuckooSet::Element k, size_t bidx) const {
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

thrust::host_vector<CuckooSet::Element> CuckooSet::Serialize() const {
  thrust::host_vector<Element> vals(NUM_BUCKETS * BinsPerBucket() * SlotsPerBin());
  auto it = vals.begin();
  for (auto& bucket : buckets_) {
    for (auto& slot : bucket) {
      it = thrust::copy(slot.begin(), slot.end(), it);
    }
  }
  return vals;
}

bool GenerateCuckooSetsFromFile(const std::string& filename,
                                double heldout_ratio,
                                std::unique_ptr<CuckooSet>* training, std::unique_ptr<CuckooSet>* heldout
                                ) {
  LOG(INFO) << "Going to generate cuckoo sets from " << filename << " with held-out ratio " << heldout_ratio;
  std::ifstream in(filename);
  std::string line;
  // skip first 4 lines
  for (int i = 0; i < 4; ++i) std::getline(in, line);
  std::vector<uint64_t> vals;
  do {
    uint64_t a, b, x, y;
    in >> a >> b;
    if (!in.eof()) {
      x = std::min(a, b);
      y = std::max(a, b);
      vals.push_back((x << 32) | y);
    }
  } while (in.good());
  if (in.bad()) {
    LOG(ERROR) << "Error reading file " << filename;
    return false;
  }
  std::sort(vals.begin(), vals.end());
  // squeeze out duplicates
  auto end = std::unique(vals.begin(), vals.end());
  vals.resize(end - vals.begin());
  // shuffle again
  std::random_shuffle(vals.begin(), vals.end());
  size_t training_len =
      static_cast<size_t>(std::ceil((1 - heldout_ratio) * vals.size()));
  size_t heldout_len = vals.size() - training_len;
  if (heldout_len > 0) {
    heldout->reset(new CuckooSet(heldout_len));
    for (auto it = vals.begin(); it != vals.begin() + heldout_len; ++it) {
      if (!(*heldout)->Insert(*it)) {
        LOG(ERROR) << "Failed to insert into heldout set, data from " << filename;
        heldout->reset();
        return false;
      }
    }
  }
  training->reset(new CuckooSet(training_len));
  for (auto it = vals.begin() + heldout_len; it != vals.end(); ++it) {
    if (!(*training)->Insert(*it)) {
      LOG(ERROR) << "Failed to insert into training set, data from " << filename;
      training->reset();
      if (heldout_len > 0) {
        heldout->reset();
      }
      return false;
    }
  }
  return true;
}

}  // namespace mcmc
