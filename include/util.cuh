#ifndef _UTIL_
#define _UTIL_

#define EPSILON (0.0001f)
#define PI (3.1515926f)

#include "vec3.cuh"
#include "vec2.cuh"
#include <thrust/random.h>

#include <curand.h>
#include <curand_kernel.h>

class vec3;
__host__ __device__ vec3 displaceOnOrtogonalPlane( vec3 dir, float2 position);
__host__ __device__ float randf();
void checkError( const char *const file, int const line );


#define cudaCheckError() checkError( __FILE__, __LINE__ )
#endif
