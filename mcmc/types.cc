#include "mcmc/types.h"

#include "mcmc/gen-util.h"

namespace mcmc {

std::string GetClTypes() {
  std::ostringstream out;
#ifdef MCMC_USE_CL
  if (type_name<Float>() == std::string("double")) {
    out << "#pragma OPENCL EXTENSION cl_khr_fp64: enable " << std::endl;
  }
  out << R"%%(
  #define KERNEL __kernel
  #define GLOBAL __global
  #define LOCAL_DECLARE __local
  #define LOCAL __local
  #define CONSTANT __constant
  #define GET_GLOBAL_ID() (get_global_id(0))
  #define GET_GLOBAL_SIZE() (get_global_size(0))
  #define GET_LOCAL_ID() (get_local_id(0))
  #define GET_LOCAL_SIZE() (get_local_size(0))
  #define GET_NUM_GROUPS() (get_num_groups(0))
  #define GET_GROUP_ID() (get_group_id(0))
  #define BARRIER_LOCAL  barrier(CLK_LOCAL_MEM_FENCE)
  #define BARRIER_GLOBAL  barrier(CLK_GLOBAL_MEM_FENCE)
  )%%";
#else
  out << R"%%(
  #define KERNEL extern "C" __global__
  #define GLOBAL
  #define LOCAL_DECLARE __shared__
  #define LOCAL
  #define CONSTANT __constant__
  #define GET_GLOBAL_ID() (blockIdx.x * blockDim.x + threadIdx.x)
  #define GET_GLOBAL_SIZE() (gridDim.x * blockDim.x)
  #define GET_LOCAL_ID() (threadIdx.x)
  #define GET_LOCAL_SIZE() (blockDim.x)
  #define GET_NUM_GROUPS() (gridDim.x)
  #define GET_GROUP_ID() (blockIdx.x)
  #define BARRIER_LOCAL  __syncthreads()
  #define BARRIER_GLOBAL __syncthreads()
  typedef unsigned int uint;
  typedef unsigned long ulong;
  #define ULONG_MAX 0xffffffffffffffffUL
  )%%";
#endif
  out << "typedef " << type_name<Float>() << " Float;" << std::endl;
  out << R"%%(
  typedef ulong uint64_t;
  typedef uint uint32_t;
  )%%";
  return gen::MakeHeader("CL_TYPES", out.str());
}

std::vector<std::string> GetClFlags(uint32_t wg) {
  std::vector<std::string> ret;
  if (wg != 0) {
    ret.push_back(std::string("-DWG_SIZE=") + std::to_string(wg));
  }
#ifdef MCMC_USE_CL
  ret.push_back("-cl-fast-relaxed-math");
#else
  ret.push_back("-default-device");
  ret.push_back("--gpu-architecture=compute_20");
  ret.push_back("-use_fast_math");
#endif
  return ret;
}

uint32_t GetMaxGroups() { return 65535; }

}  // namespace mcmc
