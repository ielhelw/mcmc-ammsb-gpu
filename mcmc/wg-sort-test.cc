#include <gtest/gtest.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/random.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/utility/source.hpp>

namespace compute = boost::compute;

const std::string kSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

    __kernel void test(__global uint* in, __global uint* out, uint len) {
      size_t id = get_local_id(0);
      out[id] = in[id];
    }

    __kernel void sort(__global uint* in, __global uint* out, uint len,
                       __local uint* aux) {
      size_t i = get_local_id(0);
      size_t wg = get_local_size(0);
      aux[i] = in[i];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (size_t length = 1; length < wg; length <<= 1) {
        bool direction = ((i & (length << 1)) != 0);
        for (size_t inc = length; inc > 0; inc >>= 1) {
          size_t j = i ^ inc;
          uint idata = aux[i];
          uint jdata = aux[j];
          bool smaller = (jdata < idata) || (jdata == idata && j < i);
          bool swap = smaller ^ (j < i) ^ direction;
          barrier(CLK_LOCAL_MEM_FENCE);
          aux[i] = swap ? jdata : idata;
          barrier(CLK_LOCAL_MEM_FENCE);
        }
      }
      out[i] = aux[i];
    }

    // Sums first 2*wg_size elements from in and stores result in out[0]
    void sum_wg(__global uint* in, __global uint* out, __local uint* aux,
                uint plen) {
      uint lid = get_local_id(0);
      uint lsize = get_local_size(0);
      uint stride = 2 * lid;
      aux[lid] = 0;
      if (stride     < plen) aux[lid] += in[stride];
      if (stride + 1 < plen) aux[lid] += in[stride + 1];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint s = lsize; s > 0; s >>= 1) {
        if (lid < s) {
          aux[lid] += aux[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (lid == 0) *out = aux[0];
    }

    // Sums values of in[i:i+2*wg_size] in out[i(/2*wg_size)]
    uint sum_partial(__global uint* in, __global uint* out, __local uint* aux,
                     uint len) {
      uint lid = get_local_id(0);
      uint stride = 2 * get_local_size(0);
      uint i;
      for (i = 0; i < len; i += stride) {
        uint plen = stride;
        if (i + plen > len) {
          plen = len - i;
        }
        uint offset = i / stride;
        sum_wg(in + i, out + offset, aux, plen);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      return i / stride;
    }

    // Sums the in[0:len] to out[0]. Out size at least ceil(len/wg_size)
    __kernel void sum(__global uint* in, __global uint* out, __local uint* aux,
                      uint len) {
      uint lid = get_local_id(0);
      uint stride = 2 * get_local_size(0);
      len = sum_partial(in, out, aux, len);
      while (len > 1) {
        len = sum_partial(out, out, aux, len);
      }
    }

    );

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vals) {
  for (size_t i = 0; i < vals.size(); ++i) {
    out << std::setw(12) << std::setfill(' ') << vals[i] << ",";
  }
  out << std::endl;
  return out;
}

#define SETUP()                                                             \
    compute::device dev = compute::system::default_device();                \
    compute::context context(dev);                                          \
    compute::command_queue queue(context, dev,                              \
                                 compute::command_queue::enable_profiling); \
    compute::program prog =                                                 \
        compute::program::create_with_source(kSource, context);             \
    try {                                                                   \
      prog.build();                                                         \
    } catch (compute::opencl_error & e) {                                   \
      LOG(FATAL) << prog.build_log();                                       \
    }                                                                       \

TEST(WorkGroup, Sum) {
  SETUP()
  std::vector<uint32_t> vals = {1,  2,  3,  4,   5,   6,    7,
                                11, 31, 32, 33,  47,  48,   49,
                                63, 64, 65, 127, 128, 1023, 11331};
  std::vector<uint32_t> work_groups = {2, 4, 16, 32, 64};
  compute::kernel kernel = prog.create_kernel("sum");
  for (auto wg : work_groups) {
    for (auto v : vals) {
      std::vector<uint32_t> host(v);
      for (uint32_t i = 0; i < host.size(); ++i) host[i] = i + 1;
      std::random_shuffle(host.begin(), host.end());
      compute::vector<uint32_t> in(host.begin(), host.end(), queue);
      compute::vector<uint32_t> out(in.size()/wg + 1, context);
      kernel.set_arg(0, in);
      kernel.set_arg(1, out);
      kernel.set_arg(2, wg * sizeof(uint32_t), 0);
      kernel.set_arg(3, static_cast<compute::uint_>(in.size()));
      auto e = queue.enqueue_1d_range_kernel(kernel, 0, wg, wg);
      e.wait();
      compute::copy(out.begin(), out.end(), host.begin(), queue);
      EXPECT_EQ((v*(v+1))/2, host[0]);
    }
  }
}

TEST(WorkGroup, Sort) {
  SETUP();
  compute::kernel kernel = prog.create_kernel("sort");
  compute::vector<compute::uint_> in(256, queue.get_context());
  compute::vector<compute::uint_> out(in.size(), queue.get_context());
  compute::mersenne_twister_engine<compute::uint_> rand(queue);
  rand.generate(in.begin(), in.end(), queue);
  std::vector<compute::uint_> host(in.size());
  compute::copy(in.begin(), in.end(), host.begin(), queue);
  std::vector<compute::uint_> host_sorted(host.begin(), host.end());
  std::sort(host_sorted.begin(), host_sorted.end());
  kernel.set_arg(0, in);
  kernel.set_arg(1, out);
  kernel.set_arg(2, static_cast<compute::uint_>(in.size()));
  kernel.set_arg(3, in.size() * sizeof(compute::uint_), nullptr);
  compute::event e =
      queue.enqueue_1d_range_kernel(kernel, 0, in.size(), in.size());
  e.wait();
  compute::copy(out.begin(), out.end(), host.begin(), queue);
  EXPECT_EQ(host_sorted, host);
}
