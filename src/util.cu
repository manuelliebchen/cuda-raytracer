#include "util.cuh"
#include <thrust/random.h>

#include <curand.h>
#include <curand_kernel.h>

#include "vec2.cuh"
#include "vec3.cuh"

namespace crt {
__host__ __device__ vec3
displaceOnOrtogonalPlane(vec3 dir, float2 position)
{
  if (dir.x == 0 && dir.y == 0 && dir.z == 0) {
    return dir;
  }
  if (position.x == 0 && position.y == 0) {
    return dir;
  }
  // If dir is up creat fitting Vectors
  vec3 up = vec3(-1, 0, 0);
  vec3 right = vec3(0, 0, -1);

  // if not up or down overwrite with fitting Vectors
  if (dir.x != 0 && dir.z != 0) {
    float horizontal_length = sqrt(dir.x * dir.x + dir.z * dir.z);
    up = (vec3(-(dir.x / horizontal_length) * dir.y,
               horizontal_length,
               -(dir.z / horizontal_length) * dir.y))
           .normalize();
    right = (vec3(dir.z, 0, -dir.x)).normalize();
    // When down invert Vectors
  } else if (dir.y < 0) {
    up *= -1;
    right *= -1;
  }

  return dir + up * position.y + right * position.x;
};

__host__ __device__ float
randf()
{
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> dist(0, 1);
  rng.discard(10);
  return dist(rng);
}

void
checkError(const char* const file, int const line)
{
  if (cudaPeekAtLastError() != cudaSuccess) {
    cudaError_t error = cudaGetLastError();
    std::cout << "Cuda Error in File: " << file << " at " << line << ":\n\t"
              << cudaGetErrorName(error) << ":\n\t\t"
              << cudaGetErrorString(error) << std::endl;
  }
};
};