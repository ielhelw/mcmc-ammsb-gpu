#include "mcmc/algorithm/normalize.h"

#include <boost/compute/utility/source.hpp>

#include "mcmc/algorithm/sum.h"
#include "mcmc/gen-util.h"

namespace mcmc {
namespace algorithm {

static const std::string kNormalizeSourceTemplate =
    BOOST_COMPUTE_STRINGIZE_SOURCE(

        __kernel void WG_NORMALIZE_TT(__global TT* in, __global TT* out,
                                      __local TT* aux, uint len) {
          size_t lid = get_local_id(0);
          size_t stride = get_local_size(0);
          size_t offset = len * get_group_id(0);
          WG_SUM_TT(in, out, aux, len);
          in += offset;
          out += offset;
          TT sum = out[0];
          for (; lid < len; lid += stride) {
            in[lid] = in[lid] / sum;
          }
        }

        );

std::string WorkGroupNormalize(const std::string& type) {
  return WorkGroupSum(type) +
         mcmc::gen::MakeHeaderFromTemplate(
             type + "_WG_NORMALIZE", kNormalizeSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
