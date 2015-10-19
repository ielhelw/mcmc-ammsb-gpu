#include "mcmc/algorithm/normalize.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sum.h"
#include "mcmc/gen-util.h"
#include "mcmc/partitioned-alloc.h"

namespace mcmc {

namespace algorithm {

static const std::string kNormalizeSourceTemplate =
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        void WG_NORMALIZE_TT(GLOBAL TT* in,
                             LOCAL TT* aux, uint len) {
          uint lid = GET_LOCAL_ID();
          uint stride = GET_LOCAL_SIZE();
          WG_SUM_TT(in, aux, len);
          TT sum = aux[0];
          for (; lid < len; lid += stride) {
            in[lid] = in[lid] / sum;
          }
        }

        KERNEL void WG_NORMALIZE_KERNEL_TT(
            GLOBAL TT* in, LOCAL TT* aux, uint len) {
          uint gid = GET_GROUP_ID();
          uint stride = GET_LOCAL_SIZE();
          in += gid * len;
          WG_NORMALIZE_TT(in, aux, len);
        }

        KERNEL void WG_NORMALIZE_PARTITIONED_KERNEL_TT(
            GLOBAL void* in, GLOBAL TT* g_sum, LOCAL TT* aux) {
          uint lsize = GET_LOCAL_SIZE();
          uint lid = GET_LOCAL_ID();
          uint gid = GET_GROUP_ID();
          GLOBAL TTRowPartitionedMatrix* pm =
              (GLOBAL TTRowPartitionedMatrix*)in;
          GLOBAL TT* row = TTRowPartitionedMatrix_Row(pm, gid);
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
