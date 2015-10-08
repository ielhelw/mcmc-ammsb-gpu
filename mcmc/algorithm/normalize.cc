#include "mcmc/algorithm/normalize.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sum.h"
#include "mcmc/gen-util.h"

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

        );

std::string WorkGroupNormalizeProgram(const std::string& type) {
  return WorkGroupSum(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_NORMALIZE",
             std::string(type == std::string("double") ? "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n" : "") +
             kNormalizeSourceTemplate,
             "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
