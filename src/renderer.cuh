#ifndef RENDERER
#define RENDERER

#include <cmath>
#include <cassert>
#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "vec3.cuh"

#include "deviceRenderer.cuh"

#include "image.cuh"
#include "deviceImage.cuh"
#include "scene.h"
#include "deviceScene.cuh"
#include "camera.cuh"

#include "util.cuh"

#include "JSON.h"
#include "JSONValue.h"

namespace crt {

class DeviceRenderer;

class Renderer {
  friend class DeviceRenderer;

  public:
    Renderer();
    Renderer(const Renderer& _renderer);
    const Renderer operator = (const Renderer& _renderer);
    ~Renderer();

    void readeSceneFile( std::string file);

    void runOnDevice();
    void runOnDeviceSingleThreadet();
    void runOnHost();

    void writeImage( std::string imageFile) const;

  private:
    Camera camera;

    Image image;
    Scene scene;

    unsigned int multisamples;
};

__global__ void globalRenderSingleThreadet( DeviceRenderer * renderer);
__global__ void globalRender( DeviceRenderer * renderer);

};

#endif
