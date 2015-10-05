#include "mcmc/algorithm/sum.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/gen-util.h"

namespace mcmc {
namespace algorithm {

static const std::string kSumSourceTemplate = BOOST_COMPUTE_STRINGIZE_SOURCE(
    // Sums first 2*wg_size elements from in and stores result in out[0]
    void WG_SUM_TT_BLOCK_(__global TT* in, __global TT* out, __local TT* aux,
                          uint plen) {
      size_t lid = get_local_id(0);
      size_t lsize = get_local_size(0);
      size_t stride = 2 * lid;
      aux[lid] = 0;
      if (stride < plen) aux[lid] += in[stride];
      if (stride + 1 < plen) aux[lid] += in[stride + 1];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint s = lsize >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          aux[lid] += aux[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lid == 0) *out = aux[0];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Sums values of in[i:i+2*wg_size] in out[i(/2*wg_size)]
    uint WG_SUM_TT_PARTIAL_(__global TT* in, __global TT* out, __local TT* aux,
                            uint len) {
      size_t lid = get_local_id(0);
      size_t stride = 2 * get_local_size(0);
      size_t i;
      for (i = 0; i < len; i += stride) {
        uint plen = stride;
        if (i + plen > len) {
          plen = len - i;
        }
        uint offset = i / stride;
        WG_SUM_TT_BLOCK_(in + i, out + offset, aux, plen);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      return i / stride;
    }

    // Sums the in[0:len] to out[0]. Out size at least ceil(len/wg_size)
    __kernel void WG_SUM_TT(__global TT* in, __global TT* out, __local TT* aux,
                            uint len) {
      size_t lid = get_local_id(0);
      size_t stride = 2 * get_local_size(0);
      size_t offset = len * get_group_id(0);
      in += offset;
      out += offset;
      len = WG_SUM_TT_PARTIAL_(in, out, aux, len);
      while (len > 1) {
        len = WG_SUM_TT_PARTIAL_(out, out, aux, len);
      }
    }

    );

std::string WorkGroupSum(const std::string& type) {
  return mcmc::gen::MakeHeaderFromTemplate(type + "_WG_SUM", kSumSourceTemplate,
                                           "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
