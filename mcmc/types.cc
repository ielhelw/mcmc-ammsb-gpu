#include "mcmc/types.h"

#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>
#include <glog/logging.h>

#include "mcmc/gen-util.h"

namespace mcmc {

std::string apply_tuple2(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg << ".x " << suffix << "),"
      << "(" << prefix << " " << arg << ".y " << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple4(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg << ".x " << suffix << "),"
      << "(" << prefix << " " << arg << ".y " << suffix << "),"
      << "(" << prefix << " " << arg << ".z " << suffix << "),"
      << "(" << prefix << " " << arg << ".w " << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple8(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg << ".lo.x " << suffix << "),"
      << "(" << prefix << " " << arg << ".lo.y " << suffix << "),"
      << "(" << prefix << " " << arg << ".lo.z " << suffix << "),"
      << "(" << prefix << " " << arg << ".lo.w " << suffix << "),"
      << "(" << prefix << " " << arg << ".hi.x " << suffix << "),"
      << "(" << prefix << " " << arg << ".hi.y " << suffix << "),"
      << "(" << prefix << " " << arg << ".hi.z " << suffix << "),"
      << "(" << prefix << " " << arg << ".hi.w " << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple16(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg << ".s0 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s1 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s2 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s3 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s4 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s5 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s6 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s7 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s8 " << suffix << "),"
      << "(" << prefix << " " << arg << ".s9 " << suffix << "),"
      << "(" << prefix << " " << arg << ".sa " << suffix << "),"
      << "(" << prefix << " " << arg << ".sb " << suffix << "),"
      << "(" << prefix << " " << arg << ".sc " << suffix << "),"
      << "(" << prefix << " " << arg << ".sd " << suffix << "),"
      << "(" << prefix << " " << arg << ".se " << suffix << "),"
      << "(" << prefix << " " << arg << ".sf " << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple(const std::string& arg, const std::string& prefix,
                         const std::string& suffix, uint32_t vlen) {
  switch (vlen) {
    case 2:
      return apply_tuple2(arg, prefix, suffix);
    case 4:
      return apply_tuple4(arg, prefix, suffix);
    case 8:
      return apply_tuple8(arg, prefix, suffix);
    case 16:
      return apply_tuple16(arg, prefix, suffix);
    default:
      LOG(FATAL) << "NOT IMPLEMENTED: " << vlen;
  }
  return "";
}

std::string apply_tuple_tuple2(
    const std::string& arg0, const std::string& arg1,
    const std::string& prefix, const std::string join,
    const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg0 << ".x " << join << " " << arg1 << ".x" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".y " << join << " " << arg1 << ".y" << suffix << ")"
      << ")";
  return out.str();
}

std::string apply_tuple_tuple4(
    const std::string& arg0, const std::string& arg1,
    const std::string& prefix, const std::string join,
    const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg0 << ".x " << join << " " << arg1 << ".x" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".y " << join << " " << arg1 << ".y" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".z " << join << " " << arg1 << ".z" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".w " << join << " " << arg1 << ".w" << suffix << ")"
      << ")";
  return out.str();
}

std::string apply_tuple_tuple8(
    const std::string& arg0, const std::string& arg1,
    const std::string& prefix, const std::string join,
    const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg0 << ".lo.x " << join << " " << arg1 << ".lo.x" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".lo.y " << join << " " << arg1 << ".lo.y" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".lo.z " << join << " " << arg1 << ".lo.z" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".lo.w " << join << " " << arg1 << ".lo.w" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".hi.x " << join << " " << arg1 << ".hi.x" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".hi.y " << join << " " << arg1 << ".hi.y" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".hi.z " << join << " " << arg1 << ".hi.z" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".hi.w " << join << " " << arg1 << ".hi.w" << suffix << ")"
      << ")";
  return out.str();
}

std::string apply_tuple_tuple16(
    const std::string& arg0, const std::string& arg1,
    const std::string& prefix, const std::string join,
    const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << " " << arg0 << ".s0 " << join << " " << arg1 << ".s0" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s1 " << join << " " << arg1 << ".s1`" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s2 " << join << " " << arg1 << ".s2" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s3 " << join << " " << arg1 << ".s3" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s4 " << join << " " << arg1 << ".s4" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s5 " << join << " " << arg1 << ".s5" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s6 " << join << " " << arg1 << ".s6" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s7 " << join << " " << arg1 << ".s7" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s8 " << join << " " << arg1 << ".s8" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".s9 " << join << " " << arg1 << ".s9" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".sa " << join << " " << arg1 << ".sa" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".sb " << join << " " << arg1 << ".sb" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".sc " << join << " " << arg1 << ".sc" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".sd " << join << " " << arg1 << ".sd" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".se " << join << " " << arg1 << ".se" << suffix << "),"
      << "(" << prefix << " " << arg0 << ".sf " << join << " " << arg1 << ".sf" << suffix << ")"
      << ")";
  return out.str();
}

std::string apply_tuple_tuple(
    const std::string& arg0, const std::string& arg1,
    const std::string& prefix, const std::string join,
    const std::string& suffix, uint32_t vlen) {
  switch (vlen) {
    case 2:
      return apply_tuple_tuple2(arg0, arg1, prefix, join, suffix);
    case 4:
      return apply_tuple_tuple4(arg0, arg1, prefix, join, suffix);
    case 8:
      return apply_tuple_tuple8(arg0, arg1, prefix, join, suffix);
    case 16:
      return apply_tuple_tuple16(arg0, arg1, prefix, join, suffix);
    default:
      LOG(FATAL) << "NOT IMPLEMENTED: " << vlen;
  }
  return "";
}

std::string apply_tuple_scalar2(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple_scalar4(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple_scalar8(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple_scalar16(const std::string& arg, const std::string& prefix,
                         const std::string& suffix) {
  std::ostringstream out;
  out << "("
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << "),"
      << "(" << prefix << arg << suffix << ")" << ")";
  return out.str();
}

std::string apply_tuple_scalar(
    const std::string& arg, const std::string& prefix, const std::string& suffix,
    uint32_t vlen) {
  switch (vlen) {
    case 2:
      return apply_tuple_scalar2(arg, prefix, suffix);
    case 4:
      return apply_tuple_scalar4(arg, prefix, suffix);
    case 8:
      return apply_tuple_scalar8(arg, prefix, suffix);
    case 16:
      return apply_tuple_scalar16(arg, prefix, suffix);
    default:
      LOG(FATAL) << "NOT IMPLEMENTED: " << vlen;
  }
  return "";
}

std::string apply_func_tuple2(const std::string& func, const std::string& arg) {
  std::ostringstream out;
  out << "("
      << func << "(" << arg << ".x),"
      << func << "(" << arg << ".y)" << ")";
  return out.str();
}

std::string apply_func_tuple4(const std::string& func, const std::string& arg) {
  std::ostringstream out;
  out << "("
      << func << "(" << arg << ".x),"
      << func << "(" << arg << ".y),"
      << func << "(" << arg << ".z),"
      << func << "(" << arg << ".w)" << ")";
  return out.str();
}

std::string apply_func_tuple8(const std::string& func, const std::string& arg) {
  std::ostringstream out;
  out << "("
      << func << "(" << arg << ".lo.x),"
      << func << "(" << arg << ".lo.y),"
      << func << "(" << arg << ".lo.z),"
      << func << "(" << arg << ".lo.w),"
      << func << "(" << arg << ".hi.x),"
      << func << "(" << arg << ".hi.y),"
      << func << "(" << arg << ".hi.z),"
      << func << "(" << arg << ".hi.w)" << ")";
  return out.str();
}

std::string apply_func_tuple16(const std::string& func, const std::string& arg) {
  std::ostringstream out;
  out << "("
      << func << "(" << arg << ".s0),"
      << func << "(" << arg << ".s1),"
      << func << "(" << arg << ".s2),"
      << func << "(" << arg << ".s3),"
      << func << "(" << arg << ".s4),"
      << func << "(" << arg << ".s5),"
      << func << "(" << arg << ".s6),"
      << func << "(" << arg << ".s7),"
      << func << "(" << arg << ".s8),"
      << func << "(" << arg << ".s9),"
      << func << "(" << arg << ".sa),"
      << func << "(" << arg << ".sb),"
      << func << "(" << arg << ".sc),"
      << func << "(" << arg << ".sd),"
      << func << "(" << arg << ".se),"
      << func << "(" << arg << ".sf)" << ")";
  return out.str();
}

std::string apply_func_tuple(const std::string& func, const std::string& arg,
                        uint32_t vlen) {
  switch (vlen) {
    case 2:
      return apply_func_tuple2(func, arg);
    case 4:
      return apply_func_tuple4(func, arg);
    case 8:
      return apply_func_tuple8(func, arg);
    case 16:
      return apply_func_tuple16(func, arg);
    default:
      LOG(FATAL) << "NOT IMPLEMENTED: " << vlen;
  }
  return "";
}

#ifdef MCMC_USE_CL
std::string make_v_macros(uint32_t vlen) {
  std::ostringstream out;
  out << "inline Float" << vlen <<" MAKEV" << vlen << "(Float s) { return (Float" << vlen << ")"
      << apply_tuple_scalar("s", "", "", vlen) << "; }" << std::endl;
  // make vector out of exact literal
  out << "#define VL" << vlen << "(s) (Float" << vlen << ")"
      << apply_tuple_scalar("s", "", "", vlen) << std::endl;

  out << "inline Float" << vlen << " vfabs" << vlen << "(const Float" << vlen << " a) { return (Float" << vlen << ")"
      << apply_func_tuple("FABS", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vsqrt" << vlen << "(const Float" << vlen << " a) { return (Float" << vlen << ")"
      << apply_func_tuple("SQRT", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vexp" << vlen << "(const Float" << vlen << " a) { return (Float" << vlen << ")"
      << apply_func_tuple("EXP", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vlog" << vlen << "(const Float" << vlen << " a) { return (Float" << vlen << ")"
      << apply_func_tuple("LOG", "a", vlen) << "; }" << std::endl;
  return out.str();
}
#else
std::string make_v_macros(uint32_t vlen) {
  std::ostringstream out;
  out << "#define VOP" << vlen << "(a, b, op) MAKE_FLOAT" << vlen
      // "(a.x op b.x, a.y op b.y)"
      << apply_tuple_tuple("a", "b", "", "op", "", vlen) << std::endl;
  out << "#define VSOP" << vlen << "(a, s, op) MAKE_FLOAT" << vlen
    //"(a.x op s, a.y op s)"
      << apply_tuple("a", "", " op s", vlen) << std::endl;
  out << "#define SVOP" << vlen << "(s, a, op) MAKE_FLOAT" << vlen
    //"(s op a.x, s op a.y)"
      << apply_tuple("a", "s op ", "", vlen) << std::endl;
  out << "inline Float" << vlen << " MAKEV" << vlen << "(Float s) { return MAKE_FLOAT" << vlen
    //"(s, s)" << std::endl;
      << apply_tuple_scalar("s", "", "", vlen) << "; }" << std::endl;
  // make vector out of exact literal
  out << "#define VL" << vlen << "(s) MAKE_FLOAT" << vlen
    //"(s, s)" << std::endl;
      << apply_tuple_scalar("s", "", "", vlen) << std::endl;

  out << "inline Float" << vlen << " vfabs" << vlen << "(const Float" << vlen << " a)  { return MAKE_FLOAT" << vlen
      << apply_func_tuple("FABS", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vsqrt" << vlen << "(const Float" << vlen << " a) { return MAKE_FLOAT" << vlen
      << apply_func_tuple("SQRT", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vexp" << vlen << "(const Float" << vlen << " a) { return MAKE_FLOAT" << vlen
      << apply_func_tuple("EXP", "a", vlen) << "; }" << std::endl;
  out << "inline Float" << vlen << " vlog" << vlen << "(const Float" << vlen << " a) { return MAKE_FLOAT" << vlen
      << apply_func_tuple("LOG", "a", vlen)  << "; }" << std::endl;
  return out.str();
}
#endif

#ifdef MCMC_USE_CL
std::string make_base_macros() {
  std::ostringstream out;
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
  out << "#define FABS fabs" << std::endl;
  out << "#define EXP exp" << std::endl;
  out << "#define SQRT sqrt" << std::endl;
  out << "#define LOG log" << std::endl;
  out << "#define POW pow" << std::endl;
  return out.str();
}
#else
std::string make_base_macros() {
  std::ostringstream out;
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
  #define atomic_cmpxchg atomicCAS
  typedef unsigned int uint;
  typedef unsigned long ulong;
  #define ULONG_MAX 0xffffffffffffffffUL
  )%%";
  out << "#define MAKE_FLOAT2 make_" << type_name<Float>() << "2" << std::endl;
  out << "#define MAKE_FLOAT4 make_" << type_name<Float>() << "4" << std::endl;
  if (type_name<Float>() == std::string("double")) {
    out << "#define FABS fabs" << std::endl;
    out << "#define EXP exp" << std::endl;
    out << "#define SQRT sqrt" << std::endl;
    out << "#define LOG log" << std::endl;
    out << "#define POW pow" << std::endl;
  } else {
    out << "#define FABS fabsf" << std::endl;
    out << "#define EXP expf" << std::endl;
    out << "#define SQRT sqrtf" << std::endl;
    out << "#define LOG logf" << std::endl;
    out << "#define POW powf" << std::endl;
  }
  return out.str();
}
#endif

std::string make_leaf_v_macros(uint32_t vlen) {
  std::string tmp = R"%%(
  #define VTTADD(a, b) VOPTT(a, b, +)
  #define VTTSUB(a, b) VOPTT(a, b, -)
  #define VTTMUL(a, b) VOPTT(a, b, *)
  #define VTTDIV(a, b) VOPTT(a, b, /)

  #define VTTSADD(a, s) VSOPTT(a, s, +)
  #define VTTSSUB(a, s) VSOPTT(a, s, -)
  #define VTTSMUL(a, s) VSOPTT(a, s, *)
  #define VTTSDIV(a, s) VSOPTT(a, s, /)

  #define SVTTADD(s, a) SVOPTT(s, a, +)
  #define SVTTSUB(s, a) SVOPTT(s, a, -)
  #define SVTTMUL(s, a) SVOPTT(s, a, *)
  #define SVTTDIV(s, a) SVOPTT(s, a, /)
    inline FloatTT operator+(const FloatTT a, const FloatTT b) { return VTTADD(a, b); }
    inline FloatTT operator-(const FloatTT a, const FloatTT b) { return VTTSUB(a, b); }
    inline FloatTT operator*(const FloatTT a, const FloatTT b) { return VTTMUL(a, b); }
    inline FloatTT operator/(const FloatTT a, const FloatTT b) { return VTTDIV(a, b); }
    inline FloatTT operator+=(FloatTT& a, const FloatTT b) { return a = a + b; }
    inline FloatTT operator-=(FloatTT& a, const FloatTT b) { return a = a - b; }
    inline FloatTT operator*=(FloatTT& a, const FloatTT b) { return a = a * b; }
    inline FloatTT operator/=(FloatTT& a, const FloatTT b) { return a = a / b; }

    inline FloatTT operator+(const FloatTT a, const Float s) { return VTTSADD(a, s); }
    inline FloatTT operator-(const FloatTT a, const Float s) { return VTTSSUB(a, s); }
    inline FloatTT operator*(const FloatTT a, const Float s) { return VTTSMUL(a, s); }
    inline FloatTT operator/(const FloatTT a, const Float s) { return VTTSDIV(a, s); }
    inline FloatTT operator+=(FloatTT& a, const Float s) { return a = a + s; }
    inline FloatTT operator-=(FloatTT& a, const Float s) { return a = a - s; }
    inline FloatTT operator*=(FloatTT& a, const Float s) { return a = a * s; }
    inline FloatTT operator/=(FloatTT& a, const Float s) { return a = a / s; }

    inline FloatTT operator+(const Float s, const FloatTT a) { return SVTTADD(s, a); }
    inline FloatTT operator-(const Float s, const FloatTT a) { return SVTTSUB(s, a); }
    inline FloatTT operator*(const Float s, const FloatTT a) { return SVTTMUL(s, a); }
    inline FloatTT operator/(const Float s, const FloatTT a) { return SVTTDIV(s, a); }
  )%%";
  return boost::replace_all_copy(tmp, "TT", std::to_string(vlen));
}

std::string GetClTypes() {
  std::ostringstream out;
  out << "#define Float " << type_name<Float>() << "" << std::endl;
  out << "#define Float2 " << type_name<Float>() << "2" << std::endl;
  out << "#define Float4 " << type_name<Float>() << "4" << std::endl;
#ifdef MCMC_USE_CL
  out << "#define Float8 " << type_name<Float>() << "8" << std::endl;
  out << "#define Float16 " << type_name<Float>() << "16" << std::endl;
#endif
  if (type_name<Float>() == std::string("double")) {
    out << "#define FL(X) (X)" << std::endl;
  } else {
    out << "#define FL(X) (X ## f)" << std::endl;
  }
  out << make_base_macros();
  out << R"%%(
  typedef ulong uint64_t;
  typedef uint uint32_t;
  )%%";
  out << make_v_macros(2);
  out << make_v_macros(4);
#ifndef MCMC_USE_CL
  out << make_leaf_v_macros(2);
  out << make_leaf_v_macros(4);
#else
  out << make_v_macros(8);
  out << make_v_macros(16);
#endif
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

std::ostream& operator<<(std::ostream& out, const ulong2& v) {
  out << v[0] << "," << v[1];
  return out;
}

std::istream& operator>>(std::istream& in, ulong2& v) {
  in >> v[0];
  if (in.get() != ',') {
    throw boost::program_options::validation_error(
        boost::program_options::validation_error::invalid_option_value,
        "Invalid ulong2");
  }
  in >> v[2];
  return in;
}

}  // namespace mcmc
