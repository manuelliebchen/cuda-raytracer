#ifndef _MATERIAL_
#define _MATERIAL_

#include <iostream>

class Material {
  public:
    Material ( float3 _color, float _diffuse, float _reflection, float _refraction, float _refractionIndex, float _emission);
    __host__ __device__ Material ( const Material& material);

    __host__ __device__ float3 getBaseColor() const;
    __host__ __device__ float getEmission() const;
    __host__ __device__ float getDiffuse() const;
    __host__ __device__ float getSpecular() const;


  private:
    float3 color;

    float diffuse;
    float reflection;
    float refraction;

    float refractionIndex;

    float emission;
};

#endif
