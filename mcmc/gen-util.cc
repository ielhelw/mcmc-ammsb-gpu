#include "mcmc/gen-util.h"

#include <boost/algorithm/string/replace.hpp>

namespace mcmc {
namespace gen {

std::string MakeHeaderFromTemplate(const std::string& guard,
                                   const std::string& source,
                                   const std::string& pattern,
                                   const std::string& replacement) {
  return MakeHeader(guard,
                    boost::replace_all_copy(source, pattern, replacement));
}

std::string MakeHeader(const std::string& guard, const std::string& source) {
  std::ostringstream out;
  out << std::endl;
  out << "#ifndef __" << guard << "__TEMPLATE__" << std::endl;
  out << "#define __" << guard << "__TEMPLATE__" << std::endl;
  out << source << std::endl;
  out << "#endif //  __" << guard << "__TEMPLATE__" << std::endl;
  return out.str();
}

}  // namespace gen
}  // namespace mcmc
