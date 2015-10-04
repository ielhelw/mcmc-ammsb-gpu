#ifndef __MCMC_CUCKOO_H__
#define __MCMC_CUCKOO_H__

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

#include "mcmc/types.h"

namespace mcmc {
namespace cuckoo {

namespace compute = boost::compute;

const std::string kClSetTypes = BOOST_COMPUTE_STRINGIZE_SOURCE(
    typedef ulong uint64_t; typedef uint uint32_t;

    typedef struct {
      __global uint64_t* base_;
      uint64_t num_bins_;
    } Set;

    );

const std::string kClSetHeader =
    kClSetTypes +
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        __constant uint64_t NUM_BUCKETS = 2; __constant uint64_t NUM_SLOTS = 4;

        uint64_t hash1(uint64_t num_bins,
                       uint64_t k) { return (1003 * k) % num_bins; }

        uint64_t hash2(uint64_t num_bins,
                       uint64_t k) { return (k ^ 179440147) % num_bins; }

        bool Set_SlotHasEdge_(__global uint64_t* slot, uint64_t k) {
          if (k == slot[0]) return true;
          if (k == slot[1]) return true;
          if (k == slot[2]) return true;
          if (k == slot[3]) return true;
          return false;
        }

        bool Set_HasEdge(__global Set* set, uint64_t k) {
          uint64_t h1 = hash1(set->num_bins_, k);
          if (Set_SlotHasEdge_(
                  set->base_ + 0 * set->num_bins_ * NUM_SLOTS + h1 * NUM_SLOTS,
                  k))
            return true;
          uint64_t h2 = hash2(set->num_bins_, k);
          if (Set_SlotHasEdge_(
                  set->base_ + 1 * set->num_bins_ * NUM_SLOTS + h2 * NUM_SLOTS,
                  k))
            return true;
          return false;
        }

        );

class Set {
 public:
  static const size_t NUM_BUCKETS = 2;
  static const size_t NUM_SLOTS = 4;

  typedef std::array<Edge, NUM_SLOTS> Slot;
  typedef std::vector<Slot> Bucket;
  typedef std::array<Bucket, NUM_BUCKETS> Buckets;

  static const Edge KEY_INVALID;

  Set(size_t n);

  bool Insert(Edge k);

  bool Has(Edge k) const;

  inline size_t BinsPerBucket() const { return N_; }

  std::vector<Edge> Serialize() const;

 private:
  bool IsSlotNotFullAndKeyNotInIt(Edge k, const Slot& slot) const;

  bool IsSlotFull(const Slot& slot) const;

  bool IsKeyInSlot(Edge k, const Slot& slot) const;

  Edge InsertKeyInSlot(Edge k, Slot* slot);

  size_t Hash(Edge k, size_t bidx) const;

  const size_t N_;
  Buckets buckets_;
  uint32_t seed_;
  const size_t displacements_max_;
};

class OpenClSetFactory;

class OpenClSet {
 public:
  inline compute::buffer& Get() { return buf_; }

 private:
  OpenClSet(std::shared_ptr<OpenClSetFactory> factory, compute::kernel* init,
            compute::command_queue* queue, uint64_t sizeOfSet,
            const std::vector<uint64_t>& data);

  std::shared_ptr<OpenClSetFactory> factory_;
  compute::command_queue queue_;
  compute::vector<uint64_t> data_;
  uint64_t num_bins_;
  compute::buffer buf_;

  friend class OpenClSetFactory;
};

class OpenClSetFactory : public std::enable_shared_from_this<OpenClSetFactory> {
 public:
  static std::shared_ptr<OpenClSetFactory> New(compute::command_queue queue);

  OpenClSet* CreateSet(const std::vector<uint64_t>& data);

 private:
  OpenClSetFactory(compute::command_queue queue);

  compute::program prog_;
  compute::command_queue queue_;
  compute::kernel init_kernel_;
  uint64_t sizeOfSet_;
};

}  // namespace cuckoo
}  // namespace mcmc

#endif  // __MCMC_CUCKOO_H__
