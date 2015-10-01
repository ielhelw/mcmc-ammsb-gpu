#ifndef __CUCKOO_H__
#define __CUCKOO_H__

#include <array>
#include <limits>
#include <memory>
#include <vector>

namespace mcmc {

class CuckooSet {
 public:
  static const size_t NUM_BUCKETS = 2;

  typedef uint64_t Element;
  typedef std::vector<Element> Slot;
  typedef std::vector<Slot> Bucket;
  typedef std::array<Bucket, NUM_BUCKETS> Buckets;

  static const Element KEY_INVALID;

  CuckooSet(size_t n, size_t num_slots = 4);

  bool Insert(Element k);

  bool Has(Element k) const;

 private:
  bool IsSlotNotFullAndKeyNotInIt(Element k, const Slot& slot) const;

  bool IsSlotFull(const Slot& slot) const;

  bool IsKeyInSlot(Element k, const Slot& slot) const;

  Element InsertKeyInSlot(Element k, Slot* slot);

  size_t Hash(Element k, size_t bidx) const;

  size_t N_;
  Buckets buckets_;
  uint32_t seed_;
  const size_t displacements_max_;
};

bool GenerateCuckooSetsFromFile(const std::string& filename,
                                double heldout_ratio, std::unique_ptr<CuckooSet>* training,
                                std::unique_ptr<CuckooSet>* heldout);

}  // namespace mcmc

#endif  // __CUCKOO_H__
