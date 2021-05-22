cmake_minimum_required(VERSION 3.9)

project( Raytracer )

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CC_STANDARD 11)

find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

FILE(GLOB_RECURSE CUDA_SOURCE ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

list(APPEND CUDA_NVCC_FLAGS 
  -arch=sm_61
  -maxrregcount=64
  #--resource-usage
  # --ptxas-options=-v
  -rdc=true
)

set( CUDA_SEPARABLE_COMPILATION ON) #Separated Classes

CUDA_ADD_EXECUTABLE( raytracer ${CUDA_SOURCE})

set_target_properties(raytracer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_SOURCE_DIR}/build)

target_include_directories(raytracer PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  ${PNG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/dep/simplejson/src
)

target_link_libraries( raytracer ${CUDA_LIBRARIES} ${PNG_LIBRARIES})

#
# Code Quality
#
OPTION(USE_CLANG_FORMAT "Formats the codebase" ON)
if(USE_CLANG_FORMAT)
  find_program(CLANG_FORMAT "clang-format")
  if(CLANG_FORMAT)
    add_custom_target(
      clang-format ALL
      COMMAND ${CLANG_FORMAT}
      -i
      -style=Mozilla
      ${CUDA_SOURCE}
      )
  else()
    message(STATUS "Could not find clang-format")
  endif()
endif()

# Use make 2> iwyu.out && fix_include <iwyu.out   
OPTION(USE_IWYU "Use include what you use for include optimization" OFF)
IF(USE_IWYU)
    find_program(IWYU include-what-you-use HINTS /usr/bin/)
    if(NOT IWYU)
        message(STATUS "IWYU not found!")
    endif()

    set_property(TARGET raytracer
        PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${IWYU} --check_also=${CUDA_SOURCE})
endif()