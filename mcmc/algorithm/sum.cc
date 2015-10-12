#include "mcmc/algorithm/sum.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/gen-util.h"
#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {
namespace algorithm {

static const std::string kSumSourceTemplate = BOOST_COMPUTE_STRINGIZE_SOURCE(

    void WG_SUM_TT_BLOCK_(__global TT* in, __global TT* scratch,
                          __local TT* aux, uint len) {
      uint lid = get_local_id(0);
      uint stride = lid << 1;
      aux[lid] = 0;
      if (stride < len) aux[lid] += in[stride];
      if (stride + 1 < len) aux[lid] += in[stride + 1];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          aux[lid] += aux[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lid == 0) *scratch = aux[0];
      barrier(CLK_GLOBAL_MEM_FENCE);
    }

    void WG_SUM_TT_PARTIAL_(__global TT* in, __global TT* scratch,
                            __local TT* aux, uint len) {
      size_t stride = get_local_size(0) << 1;
      size_t i;
      for (i = 0; i < len; i += stride) {
        uint plen = stride;
        if (i + plen > len) {
          plen = len - i;
        }
        uint offset = i / stride;
        WG_SUM_TT_BLOCK_(in + i, scratch + offset, aux, plen);
      }
    }

    void WG_SUM_FOLD_TT(__global TT* scratch, __local TT* aux, uint len) {
      uint lsizex2 = get_local_size(0) << 1;
      while (len > 1) {
        WG_SUM_TT_PARTIAL_(scratch, scratch, aux, len);
        len = len / lsizex2 + (len % lsizex2 ? 1 : 0);
      }
    }

    void WG_SUM_TT(__global TT* in, __global TT* scratch, __local TT* aux,
                   uint len) {
      uint lsizex2 = get_local_size(0) << 1;
      WG_SUM_TT_PARTIAL_(in, scratch, aux, len);
      len = len / lsizex2 + (len % lsizex2 ? 1 : 0);
      WG_SUM_FOLD_TT(scratch, aux, len);
    }

    __kernel void WG_SUM_KERNEL_TT(__global TT* in, __global TT* out,
                                   __global TT* scratch, __local TT* aux,
                                   uint len) {
      uint lsize = get_local_size(0);
      uint gid = get_group_id(0);
      uint scratch_per_wg = len / lsize + (len % lsize ? 1 : 0);
      in += len * gid;
      scratch += gid * scratch_per_wg;
      WG_SUM_TT(in, scratch, aux, len);
      if (get_local_id(0) == 0) out[gid] = *scratch;
    }

    __kernel void WG_SUM_PARTITIONED_KERNEL_TT(
        __global void* in, __global TT* out, __global TT* scratch,
        __local TT* aux) {
      uint lsize = get_local_size(0);
      uint gid = get_group_id(0);
      __global TTRowPartitionedMatrix* pm = (__global TTRowPartitionedMatrix*)in;
      uint scratch_per_wg =
          pm->num_cols_ / lsize + (pm->num_cols_ % lsize ? 1 : 0);
      __global TT* row = TTRowPartitionedMatrix_Row(pm, gid);
      scratch += gid * scratch_per_wg;
      WG_SUM_TT(row, scratch, aux, pm->num_cols_);
      if (get_local_id(0) == 0) out[gid] = *scratch;
    }

    );

std::string WorkGroupSum(const std::string& type) {
  return GetRowPartitionedMatrixHeader(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_SUM", GetClTypes() + kSumSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
