#ifndef INTERSECTION
#define INTERSECTION

#include "vec3.cuh"
#include "object.cuh"
#include "ray.cuh"

class Object;

struct Intersection {
    __host__ __device__ Intersection( vec3 _position, vec3 _normal, vec3 _in, size_t _materialIndex);
    __host__ __device__ Intersection( const Intersection& intersection);
    __host__ __device__ Intersection();
    __host__ __device__ Intersection operator= ( const Intersection& intersection);

    __host__ __device__ Ray getReflectionRay() const;
    __host__ __device__ Ray getRefrectionRay() const;

    __host__ __device__ size_t getMaterialIndex() const;

    __host__ __device__ vec3 getPosition() const;
    __host__ __device__ vec3 getNormal() const;
    __host__ __device__ vec3 getIncomming() const;

    vec3 position;
    vec3 normal;
    vec3 in;
    vec3 reflection;
    size_t materialIndex;
};

#endif
