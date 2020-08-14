#include "material.cuh"

Material::Material( float3 _color, float _diffuse, float _reflection, float _refraction, float _refractionIndex, float _emission) :
  color( _color),
  diffuse( _diffuse),
  reflection( _reflection),
  refraction( _refraction),
  refractionIndex( _refractionIndex),
  emission( _emission)
{
};

__host__ __device__
Material::Material( const Material& material) :
  color( material.color),
  diffuse( material.diffuse),
  reflection( material.reflection),
  refraction( material.refraction),
  refractionIndex( material.refractionIndex),
  emission( material.emission)
{
};


__host__ __device__
float3
Material::getBaseColor() const {
  return color;
};

__host__ __device__
float
Material::getDiffuse() const {
  return diffuse;
};

__host__ __device__
float
Material::getSpecular() const {
  return reflection;
};

__host__ __device__
float
Material::getEmission() const {
  return emission;
}
