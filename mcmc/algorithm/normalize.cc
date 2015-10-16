#include "mcmc/algorithm/normalize.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sum.h"
#include "mcmc/gen-util.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

namespace algorithm {

static const std::string kNormalizeSourceTemplate =
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void WG_NORMALIZE_TT(__global TT* in,
                             __local TT* aux, uint len) {
          uint lid = get_local_id(0);
          uint stride = get_local_size(0);
          WG_SUM_TT(in, aux, len);
          TT sum = aux[0];
          for (; lid < len; lid += stride) {
            in[lid] = in[lid] / sum;
          }
        }

        __kernel void WG_NORMALIZE_KERNEL_TT(
            __global TT* in, __local TT* aux, uint len) {
          uint gid = get_group_id(0);
          uint stride = get_local_size(0);
          in += gid * len;
          WG_NORMALIZE_TT(in, aux, len);
        }

        __kernel void WG_NORMALIZE_PARTITIONED_KERNEL_TT(
            __global void* in, __global TT* g_sum, __local TT* aux) {
          uint lsize = get_local_size(0);
          uint lid = get_local_id(0);
          uint gid = get_group_id(0);
          __global TTRowPartitionedMatrix* pm =
              (__global TTRowPartitionedMatrix*)in;
          __global TT* row = TTRowPartitionedMatrix_Row(pm, gid);
          WG_SUM_TT(row, aux, pm->num_cols_);
          Float sum = aux[0];
          for (uint i = lid; i < pm->num_cols_; i += lsize) {
            row[i] = row[i] / sum;
          }
          if (lid == 0) g_sum[gid] = sum;
        }

        );

std::string WorkGroupNormalizeProgram(const std::string& type) {
  return WorkGroupSum(type) + mcmc::gen::MakeHeaderFromTemplate(
                                  type + "_WG_NORMALIZE",
                                  GetClTypes() + kNormalizeSourceTemplate, "TT",
                                  type);
}

}  // namespace algorithm
}  // namespace mcmc
