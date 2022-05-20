#
# CMake Support
# Added by Sida Liu (learner.sida.liu@gmail.com) 2020
# 

cmake_minimum_required(VERSION 3.12.0)
project(halloc LANGUAGES CXX CUDA)

include_directories(src)
file( GLOB SRC src/*.cu )
file( GLOB SIMPLEST_SRC samples/simplest/*.cu)

add_library(halloc ${SRC})
add_executable(simplest ${SIMPLEST_SRC})
target_link_libraries(simplest halloc)

find_package(CUDA) # for CUDA_SELECT_NVCC_ARCH_FLAGS function
if (NOT DEFINED CUDA_ARCH)
  CUDA_SELECT_NVCC_ARCH_FLAGS(RET_ARCH_FLAGS)
  message("CUDA_SELECT_NVCC_ARCH_FLAGS returns: " ${RET_ARCH_FLAGS})
  list(LENGTH RET_ARCH_FLAGS FLAG_LENGTH)
  message("Flag length: " ${FLAG_LENGTH})
  if (FLAG_LENGTH GREATER 3)
    # CUDA_SELECT_NVCC_ARCH_FLAGS resulted more than 3 items, which is probably GPU not found error.
    message(FATAL_ERROR "Nvidia GPU not found. Abort.")
  endif()
  list(GET RET_ARCH_FLAGS -1 CUDA_ARCH) # use newest architecture
endif()
message("Using GPU Architecture: " ${CUDA_ARCH})
set(ARCH_FLAGS "-gencode" ${CUDA_ARCH})
list(APPEND CMAKE_CUDA_FLAGS -gencode=${CUDA_ARCH})

set_target_properties(halloc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(simplest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  if (NOT DEFINED CUDA_DEBUG)
    set(CUDA_DEBUG "-G or not" ON)
  endif()
endif(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
#-G means debug info in GPU, so you can step in GPU kernels
if (CUDA_DEBUG)
  message("Set CUDA_DEBUG to ON.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-G ${ARCH_FLAGS}>)
else (CUDA_DEBUG)
  message("Set CUDA_DEBUG to OFF.")
  set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>: ${ARCH_FLAGS}>)
endif(CUDA_DEBUG)

target_compile_options(halloc     PUBLIC ${CUDA_OPTIONS})

set(CMAKE_VERBOSE_MAKEFILE OFF) #turn on if we want to debug in compilation.
