#ifndef _VEC2_
#define _VEC2_

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/random.h>

#include "util.cuh"

struct vec2 {
  
  __host__ __device__ vec2();
  __host__ __device__ vec2( float _x, float _y);
  __host__ __device__ vec2( const vec2& other);
  __host__ __device__ vec2 operator= ( const vec2& rhs);
  __host__ __device__ ~vec2();
  
  float x;
  float y;
      
  __host__ __device__ vec2 operator+ ( const vec2& rhs) const;
  __host__ __device__ vec2 operator+= ( const vec2& rhs);
  __host__ __device__ vec2 operator- ( const vec2& rhs) const;
  __host__ __device__ vec2 operator-= ( const vec2& rhs);
  
  __host__ __device__ vec2 operator* ( float rhs) const;
  __host__ __device__ vec2 operator*= ( float rhs);
  
  __device__ operator float() const;
  __host__ __device__ float lengthSquared() const;
  
  __device__ vec2 normalize();
  __device__ vec2 normalized() const;
  
  
  __host__ __device__ static float dot( const vec2& lhs, const vec2& rhs);
  
  __host__ __device__ static vec2 random();

};

#endif //_VEC2_
