#include "mcmc/gen-util.h"

namespace mcmc {
namespace gen {

std::string MakeHeaderFromTemplate(const std::string& guard,
                                   const std::string& source,
                                   const std::string& pattern,
                                   const std::string& replacement) {
  std::regex e(pattern);
  return MakeHeader(guard, std::regex_replace(source, e, replacement));
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
