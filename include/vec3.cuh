#ifndef _VEC3_
#define _VEC3_

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/random.h>

#include "JSONValue.h"

#include "util.cuh"

struct vec3 {

  __host__ __device__ vec3();
  __host__ __device__ vec3( float _x, float _y, float _z);
  vec3( JSONArray json_vector);
  vec3( JSONValue * json_vector);
  __host__ __device__ vec3( const vec3& other);
  __host__ __device__ vec3 operator= ( const vec3& rhs);
  __host__ __device__ ~vec3();

  float x;
  float y;
  float z;

  __host__ __device__ vec3 operator+ ( const vec3& rhs) const;
  __host__ __device__ vec3 operator+= ( const vec3& rhs);
  __host__ __device__ vec3 operator- ( const vec3& rhs) const;
  __host__ __device__ vec3 operator-= ( const vec3& rhs);

  __host__ __device__ vec3 operator- () const;

  __host__ __device__ vec3 operator* ( float rhs) const;
  __host__ __device__ vec3 operator*= ( float rhs);

  __host__ __device__ operator float() const;
  __host__ __device__ float lengthSquared() const;

  __host__ __device__ vec3 normalize();
  __host__ __device__ vec3 normalized() const;


  __host__ __device__ static float dot( const vec3& lhs, const vec3& rhs);
  __host__ __device__ static vec3 cross( const vec3& lhs, const vec3& rhs);
  __host__ __device__ static vec3 reflect( const vec3& rhs, const vec3& normal);
  __host__ __device__ static vec3 refract( const vec3& rhs, const vec3& normal, const float refractionIndex);

  __host__ __device__ static vec3 random();
  __host__ __device__ static vec3 random( const vec3& normal);

  friend std::ostream& operator << (std::ostream& os, const vec3& vec);
};

std::ostream& operator << (std::ostream& os, const vec3& vec);

#endif //_VEC3_
