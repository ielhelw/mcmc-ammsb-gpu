#ifndef __MCMC_CUCKOO_H__
#define __MCMC_CUCKOO_H__

#include <array>
#include <limits>
#include <memory>
#include <vector>

namespace mcmc {

typedef uint64_t Edge;

class CuckooSet {
 public:
  static const size_t NUM_BUCKETS = 2;

  typedef std::vector<Edge> Slot;
  typedef std::vector<Slot> Bucket;
  typedef std::array<Bucket, NUM_BUCKETS> Buckets;

  static const Edge KEY_INVALID;

  CuckooSet(size_t n, size_t num_slots = 4);

  bool Insert(Edge k);

  bool Has(Edge k) const;

  inline size_t SlotsPerBin() const { return num_slots_per_bin_; }

  inline size_t BinsPerBucket() const { return N_; }

  std::vector<Edge> Serialize() const;

 private:
  bool IsSlotNotFullAndKeyNotInIt(Edge k, const Slot& slot) const;

  bool IsSlotFull(const Slot& slot) const;

  bool IsKeyInSlot(Edge k, const Slot& slot) const;

  Edge InsertKeyInSlot(Edge k, Slot* slot);

  size_t Hash(Edge k, size_t bidx) const;

  const size_t N_;
  const size_t num_slots_per_bin_;
  Buckets buckets_;
  uint32_t seed_;
  const size_t displacements_max_;
};

bool GetUniqueEdgesFromFile(const std::string& filename, std::vector<Edge>* vals);

bool GenerateCuckooSetsFromEdges(const std::vector<Edge>& vals,
                                double heldout_ratio, std::unique_ptr<CuckooSet>* training,
                                std::unique_ptr<CuckooSet>* heldout);

bool GenerateCuckooSetsFromFile(const std::string& filename,
                                double heldout_ratio, std::unique_ptr<CuckooSet>* training,
                                std::unique_ptr<CuckooSet>* heldout);

}  // namespace mcmc

#endif  // __MCMC_CUCKOO_H__
