#ifndef __MCMC_TYPES_H__
#define __MCMC_TYPES_H__

#include <limits>
#include <sstream>
#include <tuple>

#include "mcmc/types.h.inc"

#ifdef MCMC_USE_CL
#include <clpp11.h>
#else
#include <cupp11.h>
#endif

#define MCMC_TYPE_TRAIT(type, name)              \
  namespace mcmc {                               \
  namespace internal {                           \
  template <>                                    \
  struct type_name_trait<type> {                 \
    static std::string value() { return #name; } \
  };                                             \
  }                                              \
  }

namespace mcmc {

namespace clcuda = CLCudaAPI;

typedef uint64_t Edge;
typedef uint32_t Vertex;
typedef float Float;

struct ulong2 {
  uint64_t values[2];

  uint64_t& operator[](size_t i) { return values[i]; }
} __attribute__((aligned(16)));

namespace internal {

template <class T>
struct type_name_trait;

}  // namespace internal

inline std::tuple<Vertex, Vertex> Vertices(Edge e) {
  return std::make_tuple<Vertex, Vertex>(
      static_cast<Vertex>((e & 0xffffffff00000000) >> 32),
      static_cast<Vertex>((e & 0x00000000ffffffff)));
}

inline Edge MakeEdge(Vertex u, Vertex v) {
  return Edge((static_cast<Edge>(u) << 32) | static_cast<Edge>(v));
}

std::string GetClTypes();

std::vector<std::string> GetClFlags(uint32_t wg = 0);

}  // namespace mcmc

MCMC_TYPE_TRAIT(float, float)
MCMC_TYPE_TRAIT(double, double)
MCMC_TYPE_TRAIT(int32_t, int)
MCMC_TYPE_TRAIT(uint32_t, uint)
MCMC_TYPE_TRAIT(int64_t, long)
MCMC_TYPE_TRAIT(uint64_t, ulong)
MCMC_TYPE_TRAIT(ulong2, ulong2)

namespace mcmc {
template <class T>
std::string type_name() {
  return internal::type_name_trait<T>::value();
}
}  // namespace mcmc

#endif  // __MCMC_TYPES_H__
