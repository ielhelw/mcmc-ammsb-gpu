#include "mcmc/algorithm/sum.h"

#include "mcmc/gen-util.h"
#include "mcmc/types.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {
namespace algorithm {

static const std::string kSumSourceTemplate = R"%%(
    uint power_of_2(uint v) {
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      return v + 1;
    }

    void WG_SUM_TT_LOCAL_(LOCAL TT* aux, uint len) {
      uint lid = GET_LOCAL_ID();
      uint lsize = GET_LOCAL_SIZE();
      for (uint p2 = power_of_2(lsize) >> 1; p2 > 0; p2 >>= 1) {
        if (lid < p2 && lid + p2 < lsize) {
          aux[lid] += aux[lid + p2];
        }
        BARRIER_LOCAL;
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
      BARRIER_LOCAL;
      WG_SUM_TT_LOCAL_(aux, len);
    }

    KERNEL void WG_SUM_KERNEL_TT(GLOBAL TT* in, GLOBAL TT* out,
                                 uint rows, uint len) {
      LOCAL_DECLARE TT aux[WG_SIZE];
      uint gid = GET_GROUP_ID();
      for (; gid < rows; gid += GET_NUM_GROUPS()) {
        WG_SUM_TT(in + gid * len, aux, len);
        if (GET_LOCAL_ID() == 0) out[gid] = aux[0];
      }
    }

    KERNEL void WG_SUM_PARTITIONED_KERNEL_TT(
        GLOBAL void* in, GLOBAL TT* out) {
      LOCAL_DECLARE TT aux[WG_SIZE];
      uint gid = GET_GROUP_ID();
      GLOBAL TTRowPartitionedMatrix* pm =
          (GLOBAL TTRowPartitionedMatrix*)in;
      for (; gid < pm->num_rows_; gid += GET_NUM_GROUPS()) {
        GLOBAL TT* row = TTRowPartitionedMatrix_Row(pm, gid);
        WG_SUM_TT(row, aux, pm->num_cols_);
        if (GET_LOCAL_ID() == 0) out[gid] = aux[0];
      }
    }

    )%%";

std::string WorkGroupSum(const std::string& type) {
  return GetRowPartitionedMatrixHeader(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_SUM", GetClTypes() + kSumSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
