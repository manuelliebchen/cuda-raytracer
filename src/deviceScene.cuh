#ifndef _DEVICE_SCENE_
#define _DEVICE_SCENE_

#include "ray.cuh"
#include "intersection.cuh"

#include "util.cuh"

#include "material.cuh"

#include "scene.h"

#include "object.cuh"
#include "sphere.cuh"
#include "plane.cuh"

namespace crt {
class Scene;

class DeviceScene {
  friend class Scene;

  public:
    DeviceScene(const Scene& scene);
    const DeviceScene& operator=(const Scene& scene);
    ~DeviceScene();

    __device__ void copyDataToShared( char sharedMemory[], unsigned int linearThreadId );
    __device__ void copyDataToShared( char sharedMemory[] );
    __device__ bool trace( Ray& ray, Intersection& intersection) const;
    __device__ float3 phong( const Intersection& intersection) const;

    __host__ __device__ size_t getSharedMemory() const;

  private:

    char * memory;
    size_t totalMemory;

    Sphere * deviceSpheres;
    size_t numSpheres;

    Plane * devicePlanes;
    size_t numPlanes;

    Material * deviceMaterials;
    size_t numMaterials;
};
};
#endif //_DEVICE_SCENE_
