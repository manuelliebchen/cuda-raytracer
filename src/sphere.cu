#include "sphere.cuh"

namespace crt {
Sphere::Sphere(vec3 _position, size_t _material, float _radius)
  : Object(_position, _material)
  , radius(_radius){};

__host__ __device__
Sphere::Sphere(const Sphere& sphere)
  : Object(sphere.position, sphere.material)
  , radius(sphere.radius){};

__host__ __device__ bool
Sphere::intersect(Ray& ray, Intersection& intersection) const
{
  vec3 toSphere = position - ray.origin;
  float b = vec3::dot(toSphere, ray.direction);
  float disc = b * b - vec3::dot(toSphere, toSphere) + radius * radius;

  if (disc < 0) {
    return false;
  }

  disc = sqrtf(disc);
  float tnew = b - disc;
  if (tnew < 0) {
    tnew = b + disc;
  }

  if (tnew >= ((float)ray) || tnew <= EPSILON) {
    return false;
  }

  ray = tnew;
  vec3 intersectPosition = (vec3)(ray);
  vec3 normal = (intersectPosition - position).normalized();

  intersection =
    Intersection(intersectPosition, normal, ray.direction, material);
  return true;
};

__host__ __device__ vec3
Sphere::getEmissionPoint() const
{
  return position;
};
};