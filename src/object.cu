#include "object.cuh"

__host__ __device__
Object::Object( vec3 _position, size_t _material) :
  position( _position),
  material(_material)
{

};

__host__ __device__
Object::~Object(){

};

__host__ __device__
size_t
Object::getMaterialIndex() const {
  return material;
}
