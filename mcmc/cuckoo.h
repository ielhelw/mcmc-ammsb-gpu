#ifndef __MCMC_CUCKOO_H__
#define __MCMC_CUCKOO_H__

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "mcmc/types.h"

namespace mcmc {
namespace cuckoo {

using mcmc::Edge;

class Set {
 public:
  static const size_t NUM_BUCKETS = 2;
  static const size_t NUM_SLOTS = 4;

  typedef std::array<Edge, NUM_SLOTS> Slot;
  typedef std::vector<Slot> Bucket;
  typedef std::array<Bucket, NUM_BUCKETS> Buckets;

  static const Edge KEY_INVALID;

  Set(size_t n);

  bool SetContents(std::vector<Edge>::const_iterator start,
                   std::vector<Edge>::const_iterator end);

  bool Has(Edge k) const;

  inline size_t BinsPerBucket() const { return N_; }

  inline size_t Capacity() const { return N_ * NUM_SLOTS * NUM_BUCKETS; }

  inline size_t Size() const { return count_; }

  std::vector<Edge> Serialize() const;

  uint32_t PrimeIdx() const { return primeIdx_; }

 private:
  static const std::vector<std::pair<uint64_t, uint64_t>> PRIMES;

  void Reset();

  bool Insert(Edge k);

  bool IsSlotNotFullAndKeyNotInIt(Edge k, const Slot& slot) const;

  bool IsSlotFull(const Slot& slot) const;

  bool IsKeyInSlot(Edge k, const Slot& slot) const;

  Edge InsertKeyInSlot(Edge k, Slot* slot);

  size_t Hash(Edge k, size_t bidx) const;

  size_t count_;
  const size_t N_;
  Buckets buckets_;
  uint32_t seed_;
  const size_t displacements_max_;
  uint32_t primeIdx_;
};

class OpenClSetFactory;

class OpenClSet {
 public:
  inline clcuda::Buffer<char>& Get() { return buf_; }

 private:
  OpenClSet(std::shared_ptr<OpenClSetFactory> factory, clcuda::Kernel* init,
            clcuda::Queue* queue, uint64_t sizeOfSet, const Set& set);

  std::shared_ptr<OpenClSetFactory> factory_;
  clcuda::Queue queue_;
  clcuda::Buffer<Edge> data_;
  uint64_t num_bins_;
  clcuda::Buffer<char> buf_;

  friend class OpenClSetFactory;
};

class OpenClSetFactory : public std::enable_shared_from_this<OpenClSetFactory> {
 public:
  static const std::string& GetHeader();

  static std::shared_ptr<OpenClSetFactory> New(clcuda::Queue queue);

  OpenClSet* CreateSet(const Set& set);

 private:
  OpenClSetFactory(clcuda::Queue queue);

  clcuda::Queue queue_;
  clcuda::Program prog_;
  std::unique_ptr<clcuda::Kernel> init_kernel_;
  uint64_t sizeOfSet_;
};

}  // namespace cuckoo
}  // namespace mcmc

#endif  // __MCMC_CUCKOO_H__
