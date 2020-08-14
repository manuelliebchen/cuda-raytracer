#include "plane.cuh"

Plane::Plane( vec3 _position,  size_t _material, vec3 _uAxie, vec3 _vAxie) :
  Object( _position, _material),
  uAxie( _uAxie),
  vAxie( _vAxie)
{

};

__host__ __device__
Plane::Plane( const Plane& plane) :
  Object( plane.position, plane.material),
  uAxie( plane.uAxie),
  vAxie( plane.vAxie)
{

};

__host__ __device__
bool Plane::intersect( Ray& ray, Intersection& intersection) const {
 //Calculate Normal
 vec3 normal_plane = vec3::cross(uAxie, vAxie).normalized();
 if(vec3::dot(normal_plane, -ray.direction) < 0){
   normal_plane = -normal_plane;
 }

 //Calculate t multiplication Factor
 float t_new = ( vec3::dot( normal_plane, position - ray.origin)) / ( vec3::dot( normal_plane, ray.direction));

 if( t_new >= (float) ray || t_new <= EPSILON) {
   return false;
 }

 //PLANE -> RECTANGLE
 Ray new_ray = ray;
 new_ray = t_new;
 vec3 on_plane = (vec3) new_ray - position;
 float dot = vec3::dot(uAxie, on_plane) / (float) uAxie;
 if(dot > (float) uAxie || dot < -((float) uAxie)) {
   return false;
 }
 dot = vec3::dot(vAxie, on_plane) / (float) vAxie;
 if(dot > (float) vAxie || dot < -((float) vAxie)) {
   return false;
 }

 ray = new_ray;
 intersection = Intersection( (vec3) ray, normal_plane, ray.direction.normalized(), material);

 return true;
};

__host__ __device__
vec3 Plane::getEmissionPoint() const{
  return position + uAxie * (2 * randf() -1) + vAxie * (2 * randf() -1);
};
