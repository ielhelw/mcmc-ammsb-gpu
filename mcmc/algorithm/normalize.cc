#include "mcmc/algorithm/normalize.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sum.h"
#include "mcmc/gen-util.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

namespace algorithm {

static const std::string kNormalizeSourceTemplate =
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void WG_NORMALIZE_TT(__global TT* in, __global TT* scratch,
                             __local TT* aux, uint len) {
          uint lid = get_local_id(0);
          uint stride = get_local_size(0);
          WG_SUM_TT(in, scratch, aux, len);
          TT sum = scratch[0];
          for (; lid < len; lid += stride) {
            in[lid] = in[lid] / sum;
          }
        }

        __kernel void WG_NORMALIZE_KERNEL_TT(
            __global TT* in, __global TT* scratch, __local TT* aux, uint len) {
          uint gid = get_group_id(0);
          uint stride = get_local_size(0);
          uint scratch_per_wg = len / stride + (len % stride ? 1 : 0);
          in += gid * len;
          scratch += gid * scratch_per_wg;
          WG_NORMALIZE_TT(in, scratch, aux, len);
        }

        __kernel void WG_NORMALIZE_PARTITIONED_KERNEL_TT(
            __global void* in, __global TT* scratch, __local TT* aux) {
          uint lsize = get_local_size(0);
          uint gid = get_group_id(0);
          __global TTRowPartitionedMatrix* pm =
              (__global TTRowPartitionedMatrix*)in;
          uint scratch_per_wg =
              pm->num_cols_ / lsize + (pm->num_cols_ % lsize ? 1 : 0);
          __global TT* row = TTRowPartitionedMatrix_Row(pm, gid);
          scratch += gid * scratch_per_wg;
          WG_NORMALIZE_TT(row, scratch, aux, pm->num_cols_);
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
