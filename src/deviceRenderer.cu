#include "deviceRenderer.cuh"

#include <cassert> //asserts
#include <cmath>   //host math

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "vec3.cuh"

#include "renderer.cuh"

#include "image.cuh"
#include "renderer.cuh"
#include "scene.h"

namespace crt {
__constant__ int multisamples_pattern[42];

DeviceRenderer::DeviceRenderer(const Renderer& _renderer)
  : camera(_renderer.camera)
  , image(_renderer.image)
  , scene(_renderer.scene)
  , multisamples(_renderer.multisamples)
  , pattern_start(0)
{
  sharedMemory = scene.getSharedMemory() + 32 * 32 * sizeof(float3);
  if (multisamples == 4) {
    pattern_start = 2;
  } else if (multisamples == 16) {
    pattern_start = 10;
  }
  int pattern[42] = { 0,  0, -2, -6, 6,  -2, -6, 2, 2, 6,  1, 1,  -1, -3,
                      -3, 2, 4,  -1, -5, -2, 2,  5, 5, 3,  3, -5, -2, 6,
                      0,  7, -4, -6, -6, 4,  -8, 0, 7, -4, 6, 7,  -7, -8 };
  cudaMemcpyToSymbol(multisamples_pattern, (void*)pattern, 42 * sizeof(int));
};

const DeviceRenderer&
DeviceRenderer::operator=(const Renderer& _renderer)
{
  camera = _renderer.camera;
  image = _renderer.image;
  scene = _renderer.scene;
  multisamples = _renderer.multisamples;

  sharedMemory = scene.getSharedMemory() + 32 * 32 * sizeof(float3);
  if (multisamples == 4) {
    pattern_start = 2;
  } else if (multisamples == 16) {
    pattern_start = 10;
  }
  int pattern[42] = { 0,  0, -2, -6, 6,  -2, -6, 2, 2, 6,  1, 1,  -1, -3,
                      -3, 2, 4,  -1, -5, -2, 2,  5, 5, 3,  3, -5, -2, 6,
                      0,  7, -4, -6, -6, 4,  -8, 0, 7, -4, 6, 7,  -7, -8 };
  cudaMemcpyToSymbol(multisamples_pattern, (void*)pattern, 42 * sizeof(int));
  return (*this);
};

__host__ __device__ size_t
DeviceRenderer::getSharedMemory() const
{
  return sharedMemory;
};

DeviceRenderer::~DeviceRenderer(){};

__device__ void
DeviceRenderer::render()
{

  const int multi_sqrt = sqrtf(multisamples);
  unsigned int linearThreadId = threadIdx.x + threadIdx.y * blockDim.x;
  extern __shared__ char sharedMemory[];

  scene.copyDataToShared(sharedMemory, linearThreadId);
  float3* image_float_data = (float3*)&sharedMemory[scene.getSharedMemory()];

  __syncthreads();

  unsigned int position = linearThreadId % (1024 / multisamples);
  unsigned int sample_index = linearThreadId / (1024 / multisamples);
  float2 camera_position = make_float2(position % (blockDim.x / multi_sqrt),
                                       position / (blockDim.x / multi_sqrt));

  float pixel_angle = (float)(image.getWidth() + image.getHeight()) / 2;
  Ray ray = camera.getRay(make_float2(
    ((float)(camera_position.x + blockDim.x * blockIdx.x / multi_sqrt) -
     (0.5f * image.getWidth()) +
     ((float)multisamples_pattern[pattern_start + sample_index * 2]) / 16.0f) /
      pixel_angle,
    ((float)(camera_position.y + blockDim.y * blockIdx.y / multi_sqrt) -
     (0.5f * image.getHeight()) +
     ((float)multisamples_pattern[pattern_start + sample_index * 2 + 1]) /
       16.0f) /
      pixel_angle));

  Intersection intersection;
  float3 floatColor = make_float3(0.0f, 0.0f, 0.0f);
  if (scene.trace(ray, intersection)) {
    floatColor = scene.phong(intersection);
  }
  image_float_data[linearThreadId] = floatColor;

  __syncthreads();

  if (linearThreadId < 1024 / multisamples) {
    float3 floatColorSummarized = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < multisamples; ++i) {
      unsigned int position = linearThreadId + i * 1024 / multisamples;
      floatColorSummarized.x += image_float_data[position].x;
      floatColorSummarized.y += image_float_data[position].y;
      floatColorSummarized.z += image_float_data[position].z;
    }
    floatColorSummarized.x /= multisamples;
    floatColorSummarized.y /= multisamples;
    floatColorSummarized.z /= multisamples;

    uchar3 color =
      make_uchar3((unsigned char)(floatColorSummarized.x * 255.0f),
                  (unsigned char)(floatColorSummarized.y * 255.0f),
                  (unsigned char)(floatColorSummarized.z * 255.0f));
    uint2 imagePosition =
      make_uint2(camera_position.x + (blockDim.x * blockIdx.x / multi_sqrt),
                 camera_position.y + (blockDim.y * blockIdx.y / multi_sqrt));
    image.setPixel(imagePosition, color);
  }
};

__device__ void
DeviceRenderer::renderSingleThreadet()
{
  extern __shared__ char sharedMemory[];

  scene.copyDataToShared(sharedMemory);

  for (unsigned int i = 0; i < image.getHeight(); ++i) {
    for (unsigned int j = 0; j < image.getWidth(); ++j) {
      float3 floatColorSummarized = make_float3(0, 0, 0);
      for (unsigned int k = 0; k < multisamples; ++k) {
        uint2 imagePosition = make_uint2(j, i);
        float pixel_angle = (float)(image.getWidth() + image.getHeight()) / 2;
        Ray ray = camera.getRay(make_float2(
          ((imagePosition.x) - (0.5f * image.getWidth()) +
           ((float)multisamples_pattern[pattern_start + k * 2]) / 16.0f) /
            (pixel_angle),
          ((imagePosition.y) - (0.5f * image.getHeight()) +
           ((float)multisamples_pattern[pattern_start + k * 2 + 1]) / 16.0f) /
            (pixel_angle)));

        Intersection intersection;
        float3 floatColor = make_float3(0, 0, 0);
        if (scene.trace(ray, intersection)) {
          floatColor = scene.phong(intersection);
        }
        floatColorSummarized.x += floatColor.x;
        floatColorSummarized.y += floatColor.y;
        floatColorSummarized.z += floatColor.z;
      }
      floatColorSummarized.x /= multisamples;
      floatColorSummarized.y /= multisamples;
      floatColorSummarized.z /= multisamples;

      uchar3 color =
        make_uchar3((unsigned char)(floatColorSummarized.x * 255.0f),
                    (unsigned char)(floatColorSummarized.y * 255.0f),
                    (unsigned char)(floatColorSummarized.z * 255.0f));
      image.setPixel(make_uint2(j, i), color);
    }
  }
};
};