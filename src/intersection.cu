#include "intersection.cuh"

namespace crt {
__host__ __device__
Intersection::Intersection(vec3 _position,
                           vec3 _normal,
                           vec3 _in,
                           size_t _materialIndex)
  : position(_position)
  , normal(_normal.normalized())
  , in(_in.normalized())
  , materialIndex(_materialIndex)
{
  reflection = vec3::reflect(in, normal).normalized();
};

__host__ __device__
Intersection::Intersection(const Intersection& intersection)
  : position(intersection.position)
  , normal(intersection.normal.normalized())
  , in(intersection.in.normalized())
  , materialIndex(intersection.materialIndex)
{
  reflection = vec3::reflect(in, normal).normalized();
};

__host__ __device__ Intersection
Intersection::operator=(const Intersection& intersection)
{
  position = intersection.position;
  normal = intersection.normal.normalized();
  in = intersection.in.normalized();
  materialIndex = intersection.materialIndex;
  reflection = vec3::reflect(in, normal).normalized();
  return (*this);
};

__host__ __device__
Intersection::Intersection()
{
  materialIndex = 0;
};
};