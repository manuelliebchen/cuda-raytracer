#ifndef _PLANE_
#define _PLANE_

#include "vec3.cuh"
#include "object.cuh"
#include "ray.cuh"
#include "util.cuh"

#include <iostream>

class Plane : public Object {
  public:
    Plane( vec3 _position, size_t _materialfloat, vec3 _uAxie, vec3 _vAxie);
    __host__ __device__ Plane( const Plane& plane);

    __host__ __device__ bool intersect( Ray& ray, Intersection& intersection) const;
    __host__ __device__ vec3 getEmissionPoint() const;

  private:
    vec3 uAxie;
    vec3 vAxie;
};

#endif
