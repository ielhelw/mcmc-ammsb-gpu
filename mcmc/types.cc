#include "mcmc/types.h"

namespace mcmc {

std::string GetClTypes() {
  std::ostringstream out;
  if (compute::type_name<Float>() == std::string("double")) {
    out << "#pragma OPENCL EXTENSION cl_khr_fp64: enable " << std::endl;
  }
  out << "typedef " << compute::type_name<Float>() << " Float;" << std::endl;
  out << "#define KERNEL __kernel" << std::endl;
  out << "#define GLOBAL __global" << std::endl;
  out << "#define LOCAL __local" << std::endl;
  out << "#define GET_GLOBAL_ID() (get_global_id(0))" << std::endl;
  out << "#define GET_GLOBAL_SIZE() (get_global_size(0))" << std::endl;
  out << "#define GET_LOCAL_ID() (get_local_id(0))" << std::endl;
  out << "#define GET_LOCAL_SIZE() (get_local_size(0))" << std::endl;
  out << "#define GET_NUM_GROUPS() (get_num_groups(0))" << std::endl;
  out << "#define GET_GROUP_ID() (get_group_id(0))" << std::endl;
  return gen::MakeHeader("CL_TYPES", out.str());
}

}  // namespace mcmc
