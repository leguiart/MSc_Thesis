cmake_minimum_required(VERSION 3.14)
project(pyColSys CUDA CXX)
find_package(PythonLibs 3.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(pycol SHARED pycol.cu ${INCLUDE_FILES} ${SRC_FILES} )
#add_library(pycol SHARED bindings.cu)

target_link_libraries(pycol ${PYTHON_LIBRARIES} )
set_target_properties(pycol PROPERTIES PREFIX "")
