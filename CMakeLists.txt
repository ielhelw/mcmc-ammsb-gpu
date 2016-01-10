cmake_minimum_required(VERSION 2.8)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 -Wall -Wextra ")

find_package(Threads REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS
  program_options
  filesystem
  system
  iostreams
)
find_package(Protobuf REQUIRED)

OPTION(MCMC_USE_CL "Use OpenCL / else CUDA with nvcc" OFF)
IF (MCMC_USE_CL)
  find_package(OpenCL REQUIRED)
  find_package(BoostCompute CONFIG REQUIRED)
  macro(MCMC_ADD_LIBRARY)
    ADD_LIBRARY(${ARGN})
  endmacro()
  macro(MCMC_ADD_EXECUTABLE)
    ADD_EXECUTABLE(${ARGN})
  endmacro()
  SET(PERPLEXITY_EXT "")
ELSE()
  LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  find_package(CUDA REQUIRED)
  macro(MCMC_ADD_LIBRARY)
    CUDA_ADD_LIBRARY(${ARGN})
  endmacro()
  macro(MCMC_ADD_EXECUTABLE)
    CUDA_ADD_EXECUTABLE(${ARGN})
  endmacro()
  SET(PERPLEXITY_EXT "mcmc/perplexity.cu")
  SET(CUDA_LIBS cuda nvrtc)
ENDIF()

OPTION(MCMC_CALC_TRAIN_PPX "Calculate training perplexity" OFF)
OPTION(MCMC_SAMPLE_PARALLEL "Overlap neighbor sampling with rest of kernels" ON)

configure_file(mcmc/types.h.inc.in mcmc/types.h.inc)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS mcmc/protos.proto)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${BoostCompute_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${PROTOBUF_INCLUDE_DIRS}
)
link_libraries(
  ${CMAKE_THREAD_LIBS_INIT}
  ${GLOG_LIBRARIES}
  ${OPENCL_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CUDA_LIBS}
  ${PROTOBUF_LIBRARIES}
)

mcmc_add_library(mcmc SHARED
  ${PROTO_SRCS}
  mcmc/cuckoo.cc
  mcmc/data.cc
  mcmc/random.cc
  mcmc/gen-util.cc
  mcmc/algorithm/sort.cc
  mcmc/algorithm/sum.cc
  mcmc/algorithm/normalize.cc
  mcmc/phi.cc
  mcmc/beta.cc
  mcmc/perplexity.cc
  ${PERPLEXITY_EXT}
  mcmc/learner.cc
  mcmc/config.cc
  mcmc/types.cc
  mcmc/sample.cc
)

mcmc_add_executable(main main.cc)
target_link_libraries(main mcmc)

enable_testing()

mcmc_add_executable(cuckoo-test mcmc/cuckoo-test.cc)
target_link_libraries(cuckoo-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(cuckoo-test cuckoo-test)

mcmc_add_executable(random-test mcmc/random-test.cc)
target_link_libraries(random-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(random-test random-test)

mcmc_add_executable(data-test mcmc/data-test.cc)
target_link_libraries(data-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(data-test data-test)

mcmc_add_executable(wg-sort-test mcmc/wg-sort-test.cc)
target_link_libraries(wg-sort-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-sort-test wg-sort-test)

mcmc_add_executable(wg-sum-test mcmc/wg-sum-test.cc)
target_link_libraries(wg-sum-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-sum-test wg-sum-test)

mcmc_add_executable(wg-normalize-test mcmc/wg-normalize-test.cc)
target_link_libraries(wg-normalize-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-normalize-test wg-normalize-test)

mcmc_add_executable(wg-perplexity-test mcmc/wg-perplexity-test.cc)
target_link_libraries(wg-perplexity-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-perplexity-test wg-perplexity-test)

mcmc_add_executable(wg-phi-test mcmc/wg-phi-test.cc)
target_link_libraries(wg-phi-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-phi-test wg-phi-test)

mcmc_add_executable(wg-beta-test mcmc/wg-beta-test.cc)
target_link_libraries(wg-beta-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-beta-test wg-beta-test)

mcmc_add_executable(test-partitioned-alloc mcmc/test-partitioned-alloc.cc)
target_link_libraries(test-partitioned-alloc mcmc ${GTEST_BOTH_LIBRARIES})
add_test(test-partitioned-alloc test-partitioned-alloc)

mcmc_add_executable(wg-sample-test mcmc/wg-sample-test.cc)
target_link_libraries(wg-sample-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(wg-sample-test wg-sample-test)

mcmc_add_executable(serialize-test mcmc/serialize-test.cc)
target_link_libraries(serialize-test mcmc ${GTEST_BOTH_LIBRARIES})
add_test(serialize-test serialize-test)
