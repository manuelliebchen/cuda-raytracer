# CMakeLists.txt
cmake_minimum_required(VERSION 3.9)

project( Raytracer )

find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

set( SOURCE
  ${SRC_DIR}/JSONValue.cpp
  ${SRC_DIR}/JSON.cpp
)

set(CUDA_SOURCE
  ${SRC_DIR}/main.cu

  ${SRC_DIR}/renderer.cu
  ${SRC_DIR}/deviceRenderer.cu
  ${SRC_DIR}/image.cu
  ${SRC_DIR}/deviceImage.cu
  ${SRC_DIR}/camera.cu

  ${SRC_DIR}/scene.cu
  ${SRC_DIR}/deviceScene.cu
  ${SRC_DIR}/object.cu
  ${SRC_DIR}/material.cu

  ${SRC_DIR}/sphere.cu
  ${SRC_DIR}/plane.cu

  ${SRC_DIR}/ray.cu
  ${SRC_DIR}/intersection.cu
  ${SRC_DIR}/util.cu
  ${SRC_DIR}/vec2.cu
  ${SRC_DIR}/vec3.cu
)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PNG_INCLUDE_DIRS}
)

CUDA_INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/include
)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_61") #Pascal Architecture
list(APPEND CUDA_NVCC_FLAGS "-maxrregcount=64")
#list(APPEND CUDA_NVCC_FLAGS "--resource-usage")
list(APPEND CUDA_NVCC_FLAGS "-O2")
#list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")

list(APPEND CMAKE_CXX_FLAGS "-O2")

set( CUDA_SEPARABLE_COMPILATION ON) #Separated Classes

CUDA_ADD_EXECUTABLE( raytracer ${SOURCE} ${CUDA_SOURCE})

target_link_libraries( raytracer ${CUDA_LIBRARIES} ${PNG_LIBRARIES})
if( UNIX)
  install( TARGETS raytracer DESTINATION ~/bin)
endif( UNIX)
