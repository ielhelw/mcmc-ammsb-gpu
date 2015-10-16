
DAS5:
OpenCL_INCPATH=$OPENCL_INCLUDE OpenCL_LIBPATH=$OPENCL_LIB BoostCompute_DIR=~/opt/compute/ \
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` \
  -DGTEST_ROOT=~/opt/gtest \
  -DGLOG_ROOT=~/opt/glog \
 ..


DAS4:
OpenCL_INCPATH=$OPENCL_INCLUDE OpenCL_LIBPATH=$OPENCL_LIB BoostCompute_DIR=~/opt/compute/ \
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=`which gcc` -DCMAKE_CXX_COMPILER=`which g++` \
  -DGTEST_ROOT=~/opt/gtest \
  -DGLOG_ROOT=~/opt/glog \
  -DBOOST_ROOT=/cm/shared/package/gcc/4.9.0/ \
  -DBoost_NO_BOOST_CMAKE=ON \
  ..

