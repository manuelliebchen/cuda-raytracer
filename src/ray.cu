#include "ray.cuh"

__host__ __device__
Ray::Ray( vec3 _origin, vec3 _direcition) :
  origin(_origin),
  direction(_direcition.normalized()),
  t( FLT_MAX)
{

};

__host__ __device__
Ray::Ray( vec3 _origin, vec3 _direcition, float _t) :
  origin(_origin),
  direction(_direcition.normalized()),
  t( _t)
{

};

__host__ __device__
Ray::Ray( ) :
  t( FLT_MAX)
{
};

__host__ __device__
Ray::~Ray() {

};

__host__ __device__
Ray::Ray ( const Ray& ray) :
  origin( ray.origin),
  direction( ray.direction.normalized()),
  t( ray.t)
{

};

__host__ __device__
Ray Ray::operator= ( const Ray& ray)
{
  origin = ray.origin;
  direction = ray.direction.normalized();
  t = ray.t;
  return *this;
};

__host__ __device__
Ray Ray::operator= (float tnew) {
  t = tnew;
  return *this;
};

__host__ __device__
bool Ray::operator> (float tnew) {
  return t > tnew;
};

__host__ __device__
bool Ray::operator< (float tnew) {
  return t < tnew;
};

__host__ __device__
Ray::operator vec3() const {
  return origin + direction * t;
};

__host__ __device__
Ray::operator float() const {
  return t;
};
