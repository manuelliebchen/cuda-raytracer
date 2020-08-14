#ifndef _RAY_
#define _RAY_

#include "vec3.cuh"
#include <float.h>

struct Ray {
    __host__ __device__ Ray( vec3 _position, vec3 _direction);
    __host__ __device__ Ray( vec3 _position, vec3 _direction, float _t);
    __host__ __device__ Ray();
    __host__ __device__ ~Ray();
    __host__ __device__ Ray(const Ray& ray);
    __host__ __device__ Ray operator= (const Ray& ray);

    __host__ __device__ Ray operator= (float tnew);
    __host__ __device__ bool operator> (float tnew);
    __host__ __device__ bool operator< (float tnew);

    __host__ __device__ operator vec3() const;
    __host__ __device__ operator float() const;

    vec3 origin;
    vec3 direction;
    float t;
};

#endif
