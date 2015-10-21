#ifndef __MCMC_TYPES_H__
#define __MCMC_TYPES_H__

#include <boost/compute/type_traits/type_name.hpp>
#include <limits>
#include <sstream>
#include <tuple>

#define USE_CL

#ifdef USE_CL
#include <clpp11.h>
#else
#include <cupp11.h>
#endif

namespace mcmc {

namespace compute = boost::compute;
namespace clcuda = CLCudaAPI;

typedef uint64_t Edge;
typedef uint32_t Vertex;
typedef double Float;

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

#endif  // __MCMC_TYPES_H__
