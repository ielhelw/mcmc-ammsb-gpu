cmake_minimum_required(VERSION 2.8)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Glog REQUIRED)
find_package(CUDA REQUIRED)

#LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 -Wall -Wextra ")
#LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11  ")
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 ")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLOG_INCLUDE_DIRS}
)
link_libraries(
  ${GLOG_LIBRARIES}
)


CUDA_ADD_EXECUTABLE(main
  main.cu
  mcmc/cuckoo.cc
)
