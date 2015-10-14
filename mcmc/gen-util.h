#ifndef __MCMC_GEN_UTIL_H__
#define __MCMC_GEN_UTIL_H__

#include <sstream>

namespace mcmc {
namespace gen {

std::string MakeHeaderFromTemplate(const std::string& guard,
                                   const std::string& source,
                                   const std::string& pattern,
                                   const std::string& replacement);

std::string MakeHeader(const std::string& guard, const std::string& source);

}  // namespace gen
}  // namespace mcmc

#endif  // __MCMC_GEN_UTIL_H__
