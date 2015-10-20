#include "mcmc/algorithm/sum.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/gen-util.h"
#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {
namespace algorithm {

static const std::string kSumSourceTemplate = R"%%(
    void WG_SUM_TT_LOCAL_(LOCAL TT* aux, uint len) {
      uint lid = GET_LOCAL_ID();
      for (uint s = GET_LOCAL_SIZE() >> 1; s > 0; s >>= 1) {
        if (lid < s) {
          aux[lid] += aux[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    void WG_SUM_TT(GLOBAL TT* in, LOCAL TT* aux,
                   uint len) {
      uint lid = GET_LOCAL_ID();
      uint lsize = GET_LOCAL_SIZE();
      TT lsum = 0;
      for (uint i = lid; i < len; i += lsize) {
        lsum += in[i];
      }
      aux[lid] = lsum;
      barrier(CLK_LOCAL_MEM_FENCE);
      WG_SUM_TT_LOCAL_(aux, len);
    }

    KERNEL void WG_SUM_KERNEL_TT(GLOBAL TT* in, GLOBAL TT* out,
                                   LOCAL TT* aux,
                                   uint len) {
      uint gid = GET_GROUP_ID();
      in += len * gid;
      WG_SUM_TT(in, aux, len);
      if (GET_LOCAL_ID() == 0) out[gid] = aux[0];
    }

    KERNEL void WG_SUM_PARTITIONED_KERNEL_TT(
        GLOBAL void* in, GLOBAL TT* out,
        LOCAL TT* aux) {
      uint lsize = GET_LOCAL_SIZE();
      uint gid = GET_GROUP_ID();
      GLOBAL TTRowPartitionedMatrix* pm =
          (GLOBAL TTRowPartitionedMatrix*)in;
      GLOBAL TT* row = TTRowPartitionedMatrix_Row(pm, gid);
      WG_SUM_TT(row, aux, pm->num_cols_);
      if (GET_LOCAL_ID() == 0) out[gid] = aux[0];
    }

    )%%";

std::string WorkGroupSum(const std::string& type) {
  return GetRowPartitionedMatrixHeader(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_SUM", GetClTypes() + kSumSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
