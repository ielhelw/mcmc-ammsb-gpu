#ifndef __MCMC_TYPES_H__
#define __MCMC_TYPES_H__

#include <boost/compute/type_traits/type_name.hpp>
#include <limits>
#include <tuple>

namespace mcmc {

namespace compute = boost::compute;

typedef uint64_t Edge;
typedef uint32_t Vertex;
typedef float Float;

inline std::tuple<Vertex, Vertex> Vertices(Edge e) {
  return std::make_tuple<Vertex, Vertex>(
      static_cast<Vertex>((e & 0xffff0000) >> 32),
      static_cast<Vertex>((e & 0x0000ffff)));
}

inline Edge MakeEdge(Vertex u, Vertex v) {
  return Edge((static_cast<Edge>(u) << 32) | static_cast<Edge>(v));
}

}  // namespace mcmc

#endif  // __MCMC_TYPES_H__
