#ifndef _SPHERE_
#define _SPHERE_

#include "vec3.cuh"
#include "object.cuh"
#include "ray.cuh"

#include <iostream>

namespace crt {
class Sphere : public Object {
  public:
    Sphere( vec3 _position, size_t _materialfloat, float _radius);
    __host__ __device__ Sphere( const Sphere& sphere);

    __host__ __device__ bool intersect( Ray& ray, Intersection& intersection) const;
    __host__ __device__ vec3 getEmissionPoint() const;

  private:
    float radius;
};
};
#endif
