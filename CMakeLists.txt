cmake_minimum_required(VERSION 2.8)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Threads REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS
  program_options
  filesystem
  system
)
find_package(BoostCompute CONFIG REQUIRED)

LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 -Wall -Wextra ")
#LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 ")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${BoostCompute_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
)
link_libraries(
  ${CMAKE_THREAD_LIBS_INIT}
  ${GLOG_LIBRARIES}
  ${OPENCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_library(mcmc SHARED
  mcmc/cuckoo.cc
  mcmc/data.cc
  mcmc/random.cc
)

add_executable(cuckoo-test mcmc/cuckoo-test.cc)
target_link_libraries(cuckoo-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(cuckoo-test cuckoo-test)
