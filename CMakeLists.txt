cmake_minimum_required(VERSION 2.8)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Glog REQUIRED)
find_package(OpenCL REQUIRED)
#find_package(Boost REQUIRED)
find_package(BoostCompute CONFIG REQUIRED)

#LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 -Wall -Wextra ")
LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 ")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${BoostCompute_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
)
link_libraries(
  ${GLOG_LIBRARIES}
  ${OPENCL_LIBRARIES}
)


add_executable(main
  main.cc
  mcmc/cuckoo.cc
)
