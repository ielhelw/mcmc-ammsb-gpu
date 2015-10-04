#ifndef __MCMC_TYPES_H__
#define __MCMC_TYPES_H__

#include <limits>

typedef uint64_t Edge;
typedef uint32_t Vertex;

inline std::tuple<Vertex, Vertex> Vertices(Edge e) {
  return std::make_tuple<Vertex, Vertex>(
      static_cast<Vertex>((e & 0xffff0000) >> 32),
      static_cast<Vertex>((e & 0x0000ffff)));
}

inline Edge MakeEdge(Vertex u, Vertex v) {
  return Edge((static_cast<Edge>(u) << 32) | static_cast<Edge>(v));
}


#endif  // __MCMC_TYPES_H__
