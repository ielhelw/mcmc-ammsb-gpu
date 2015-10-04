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

    );

template <class T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vals) {
  for (size_t i = 0; i < vals.size(); ++i) {
    out << std::setw(12) << std::setfill(' ') << vals[i] << ",";
  }
  out << std::endl;
  return out;
}

TEST(WorkGroup, Sort) {
  compute::device dev = compute::system::default_device();
  compute::context context(dev);
  compute::command_queue queue(context, dev,
                               compute::command_queue::enable_profiling);
  compute::program prog =
      compute::program::create_with_source(kSource, context);
  try {
    prog.build();
  } catch (compute::opencl_error& e) {
    LOG(FATAL) << prog.build_log();
  }
  compute::kernel kernel = prog.create_kernel("sort");
  LOG(INFO) << "LOCAL MEM" << dev.local_memory_size() << ", local in kerel " << kernel.get_work_group_info<size_t>(dev, (cl_kernel_work_group_info)CL_KERNEL_LOCAL_MEM_SIZE);
  compute::vector<compute::uint_> in(
      (dev.local_memory_size() / sizeof(compute::uint_)),
      queue.get_context());
  LOG(INFO) << "USING NAX SIZE = " << in.size();
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
  LOG(INFO) << "LOCAL MEM" << dev.local_memory_size() << ", local in kerel " << kernel.get_work_group_info<size_t>(dev, (cl_kernel_work_group_info)CL_KERNEL_LOCAL_MEM_SIZE);
  compute::event e =
      queue.enqueue_1d_range_kernel(kernel, 0, in.size(), in.size());
  e.wait();
  compute::copy(out.begin(), out.end(), host.begin(), queue);
  EXPECT_EQ(host_sorted, host);
}
