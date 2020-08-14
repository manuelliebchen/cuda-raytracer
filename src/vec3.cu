#include "vec3.cuh"

__host__ __device__ vec3::vec3() :
  x(0),
  y(0),
  z(0)
{};

__host__ __device__ vec3::vec3( float _x, float _y, float _z) :
  x(_x),
  y(_y),
  z(_z)
{};

vec3::vec3( JSONArray json_vector) :
  x( json_vector[0]->AsNumber()),
  y( json_vector[1]->AsNumber()),
  z( json_vector[2]->AsNumber())
{};

vec3::vec3( JSONValue * json_vector)
{
  JSONArray json_array = json_vector->AsArray();
  x = json_array[0]->AsNumber();
  y = json_array[1]->AsNumber();
  z = json_array[2]->AsNumber();
};

__host__ __device__ vec3::vec3( const vec3& othre) :
  x(othre.x),
  y(othre.y),
  z(othre.z)
{};

__host__ __device__ vec3 vec3::operator= ( const vec3& rhs)
{
  x = rhs.x;
  y = rhs.y;
  z = rhs.z;
  return (*this);
};

__host__ __device__ vec3::~vec3()
{};

__host__ __device__ vec3 vec3::operator+ ( const vec3& rhs) const
{
  return vec3(x + rhs.x, y + rhs.y, z + rhs.z);
};

__host__ __device__ vec3 vec3::operator+= ( const vec3& rhs)
{
  x += rhs.x;
  y += rhs.y;
  z += rhs.z;
  return *this;
};

__host__ __device__ vec3 vec3::operator- ( const vec3& rhs) const
{
  return vec3(x - rhs.x, y - rhs.y, z - rhs.z);
};

__host__ __device__ vec3 vec3::operator-= ( const vec3& rhs)
{
  x -= rhs.x;
  y -= rhs.y;
  z -= rhs.z;
  return *this;
};

__host__ __device__ vec3 vec3::operator- () const
{
  return vec3( -x, -y, -z);
};


__host__ __device__ vec3 vec3::operator* ( float rhs) const
{
  return vec3( x * rhs, y * rhs, z * rhs);
};

__host__ __device__ vec3 vec3::operator*= ( float rhs)
{
  x *= rhs;
  y *= rhs;
  z *= rhs;
  return *this;
};

__host__ __device__ vec3::operator float() const
{
  return sqrtf( x*x + y*y + z*z);
};

__host__ __device__ float vec3::lengthSquared() const
{
  return x*x + y*y + z*z;
};

__host__ __device__ vec3 vec3::normalize() {
  float lenght = (float) (*this);
  x /= lenght;
  y /= lenght;
  z /= lenght;
  return *this;
};

__host__ __device__ vec3 vec3::normalized() const {
  float lenght = lengthSquared();
  if(lenght == 1){
    return vec3(*this);
  }
  lenght = sqrtf(lenght);
  return vec3( x / lenght, y / lenght, z / lenght);
};

__host__ __device__ float vec3::dot( const vec3& lhs, const vec3& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
};

__host__ __device__ vec3 vec3::cross( const vec3& lhs, const vec3& rhs) {
  return vec3( lhs.y * rhs.z - lhs.z * rhs.y,  lhs.z * rhs.x - lhs.x * rhs.z,  lhs.x * rhs.y - lhs.y * rhs.x);
};

__host__ __device__ vec3 vec3::reflect( const vec3& rhs, const vec3& normal) {
  return rhs - normal * (2 * vec3::dot(rhs, normal));
};

__host__ __device__ vec3 vec3::refract( const vec3& rhs, const vec3& normal, float refractionIndex) {
  //TODO
  return vec3();
};


__host__ __device__ vec3 vec3::random() {
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<float> uniDist( 0, 1);
    float phi = acosf(2 * uniDist(randEng) - 1);
    float lambda = uniDist(randEng) * 2 * PI;

    return vec3( cosf(lambda) * cosf(phi), sinf(phi), sinf(lambda) * cosf(phi));
};

__host__ __device__ vec3 vec3::random( const vec3& normal) {
  vec3 random = vec3::random();
  if( vec3::dot( random, normal) < 0) {
    random = -random;
  }
  return random;
};

std::ostream& operator << (std::ostream& os, const vec3& vec) {
  os << "vec3: x=" << std::to_string(vec.x) << " y=" << std::to_string(vec.y) << " z=" << std::to_string(vec.z);
  return os;
}
