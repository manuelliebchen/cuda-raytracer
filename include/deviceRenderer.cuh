#ifndef _DEVICE_RENDERER_
#define _DEVICE_RENDERER_

#include <cmath> //host math
#include <cassert> //asserts
#include <iostream> //console prints

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "vec3.cuh"

#include "image.cuh"
#include "deviceImage.cuh"

#include "scene.cuh"
#include "deviceScene.cuh"

#include "camera.cuh"

#include "renderer.cuh"

class Renderer;

class DeviceRenderer {
  friend class Renderer;

  public:
    DeviceRenderer(const Renderer& _renderer);
    const DeviceRenderer& operator = (const Renderer& _renderer);
    ~DeviceRenderer();

    __device__ void render();
    __device__ void renderSingleThreadet();

    __host__ __device__ size_t getSharedMemory() const;

  private:
    Camera camera;

    DeviceImage image;
    DeviceScene scene;

    unsigned int multisamples;

    unsigned int pattern_start;

    size_t sharedMemory;
};

#endif //_DEVICE_RENDERER_
