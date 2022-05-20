cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(CollisionDetectionSystem CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
SET(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#SET(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB INCLUDE_FILES ${PROJECT_SOURCE_DIR}/include/*.cuh)
file(GLOB TEST_FILES ${PROJECT_SOURCE_DIR}/tests/*)

add_executable(CollisionDetection ${PROJECT_SOURCE_DIR}/main.cu ${SRC_FILES} ${INCLUDE_FILES})
target_link_libraries(CollisionDetection Thrust)

# python bindings.
#add_subdirectory(bindings)

# unit tests...

# unit tests. -- need to disable FMA to get identical results on CPU and GPU.
# FMA is faster and more accurate so we leave it on in release GPU verison.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false")

add_executable(UnitTests ${TEST_FILES} ${INCLUDE_FILES}  ${SRC_FILES})
target_link_libraries(UnitTests Thrust)