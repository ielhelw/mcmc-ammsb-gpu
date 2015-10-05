#include "mcmc/gen-util.h"


namespace mcmc {
namespace gen {

std::string MakeHeaderFromTemplate(const std::string& guard,
                                   const std::string& source,
                                   const std::string& pattern,
                                   const std::string& replacement) {
  std::regex e(pattern);
  std::ostringstream out;
  out << "#ifndef __" << guard << "__TEMPLATE__" << std::endl;
  out << "#define __" << guard << "__TEMPLATE__" << std::endl;
  out << std::regex_replace(source, e, replacement) << std::endl;
  out << "#endif  __" << guard << "__TEMPLATE__" << std::endl;
  return out.str();
}

}  // namespace gen
}  // namespace mcmc
