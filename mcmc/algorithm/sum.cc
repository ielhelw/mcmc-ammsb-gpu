#include "mcmc/algorithm/sum.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/gen-util.h"
#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {
namespace algorithm {

static const std::string kSumSourceTemplate = BOOST_COMPUTE_STRINGIZE_SOURCE(
    void WG_SUM_TT_LOCAL_(__local TT* aux, uint len) {
      uint lid = get_local_id(0);
      for (uint s = get_local_size(0) >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          aux[lid] += aux[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    void WG_SUM_TT(__global TT* in, __local TT* aux,
                   uint len) {
      uint lid = get_local_id(0);
      uint lsize = get_local_size(0);
      TT lsum = 0;
      for (uint i = lid; i < len; i += lsize) {
        lsum += in[i];
      }
      aux[lid] = lsum;
      barrier(CLK_LOCAL_MEM_FENCE);
      WG_SUM_TT_LOCAL_(aux, len);
    }

    __kernel void WG_SUM_KERNEL_TT(__global TT* in, __global TT* out,
                                   __local TT* aux,
                                   uint len) {
      uint gid = get_group_id(0);
      in += len * gid;
      WG_SUM_TT(in, aux, len);
      if (get_local_id(0) == 0) out[gid] = aux[0];
    }

    __kernel void WG_SUM_PARTITIONED_KERNEL_TT(
        __global void* in, __global TT* out,
        __local TT* aux) {
      uint lsize = get_local_size(0);
      uint gid = get_group_id(0);
      __global TTRowPartitionedMatrix* pm =
          (__global TTRowPartitionedMatrix*)in;
      __global TT* row = TTRowPartitionedMatrix_Row(pm, gid);
      WG_SUM_TT(row, aux, pm->num_cols_);
      if (get_local_id(0) == 0) out[gid] = aux[0];
    }

    );

std::string WorkGroupSum(const std::string& type) {
  return GetRowPartitionedMatrixHeader(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_SUM", GetClTypes() + kSumSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
