#ifndef _DEVICE_RENDERER_
#define _DEVICE_RENDERER_

#include "camera.cuh"
#include "deviceScene.cuh"
#include "deviceImage.cuh"

namespace crt {
class Renderer;

class DeviceRenderer {

  public:
    DeviceRenderer(const Renderer& _renderer);
    const DeviceRenderer& operator= (const Renderer& _renderer);
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
    friend class Renderer;
};
};

#endif //_DEVICE_RENDERER_
