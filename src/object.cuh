#ifndef _OBJECT_
#define _OBJECT_

#include "vec3.cuh"

#include "intersection.cuh"
#include "ray.cuh"
#include "material.cuh"

namespace crt {
class Intersection;

class Object {
  public:
    __host__ __device__ virtual bool intersect( Ray& ray, Intersection& intersection) const = 0;
    __host__ __device__ virtual vec3 getEmissionPoint() const = 0;

    __host__ __device__ size_t getMaterialIndex() const;

  protected:
    __host__ __device__ Object( vec3 _position, size_t _material);
    __host__ __device__ virtual ~Object();

    vec3 position;
    size_t material;
};
};

#endif
