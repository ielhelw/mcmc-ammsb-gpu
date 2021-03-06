#include "mcmc/algorithm/sum.h"

#include "mcmc/gen-util.h"
#include "mcmc/types.h"

namespace mcmc {
namespace algorithm {

const std::string kSortSourceTemplate = R"%%(

    KERNEL void WG_SORT_TT(GLOBAL TT* in, GLOBAL TT* out, uint len) {
      LOCAL_DECLARE TT aux[WG_SIZE];
      size_t i = GET_LOCAL_ID();
      size_t wg = GET_LOCAL_SIZE();
      aux[i] = in[i];
      BARRIER_LOCAL;
      for (size_t length = 1; length < wg; length <<= 1) {
        bool direction = ((i & (length << 1)) != 0);
        for (size_t inc = length; inc > 0; inc >>= 1) {
          size_t j = i ^ inc;
          TT idata = aux[i];
          TT jdata = aux[j];
          bool smaller = (jdata < idata) || (jdata == idata && j < i);
          bool swap = smaller ^ (j < i) ^ direction;
          BARRIER_LOCAL;
          aux[i] = swap ? jdata : idata;
          BARRIER_LOCAL;
        }
      }
      out[i] = aux[i];
    }

    )%%";

std::string WorkGroupSort(const std::string& type) {
  return GetClTypes() + mcmc::gen::MakeHeaderFromTemplate(
                            type + "_WG_SORT", kSortSourceTemplate, "TT", type);
}

}  // namespace algorithm
}  // namespace mcmc
