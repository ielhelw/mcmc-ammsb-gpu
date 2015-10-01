cmake_minimum_required(VERSION 2.8)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(Glog REQUIRED)

LIST(APPEND CMAKE_CXX_FLAGS " -std=c++11 -Wall -Wextra ")

add_executable(main
  main.cc
  mcmc/cuckoo.cc
)
target_include_directories(main PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GLOG_INCLUDE_DIRS}
)
target_link_libraries(main
  ${GLOG_LIBRARIES}
)
