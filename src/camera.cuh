#ifndef CAMERA
#define CAMERA

#include "vec3.cuh"

#include "util.cuh"
#include "ray.cuh"

#include <cuda.h>
#include <iostream>

namespace crt {
class Camera {
  public:
    Camera( vec3 position, vec3 direction);
    Camera();
    Camera( const Camera& _camera);
    Camera operator= (const Camera& _camera);
    ~Camera();

    __host__ __device__ Ray getRay( float2 position);

  private:
    vec3 position;
    vec3 direction;
};
};
#endif
