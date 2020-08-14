#include "vec2.cuh"

__host__ __device__ vec2::vec2() :
  x(0),
  y(0)
{};

__host__ __device__ vec2::vec2( float _x, float _y) :
  x(_x),
  y(_y)
{};

__host__ __device__ vec2::vec2( const vec2& othre) :
  x(othre.x),
  y(othre.y)
{};

__host__ __device__ vec2 vec2::operator= ( const vec2& rhs)
{
  x = rhs.x;
  y = rhs.y;
  return (*this);
};

__host__ __device__ vec2::~vec2() 
{};

__host__ __device__ vec2 vec2::operator+ ( const vec2& rhs) const
{
  return vec2(x + rhs.x, y + rhs.y);
};

__host__ __device__ vec2 vec2::operator+= ( const vec2& rhs)
{
  x += rhs.x; 
  y += rhs.y; 
  return *this; 
};

__host__ __device__ vec2 vec2::operator- ( const vec2& rhs) const
{
  return vec2(x - rhs.x, y - rhs.y);
};

__host__ __device__ vec2 vec2::operator-= ( const vec2& rhs)
{
  x -= rhs.x; 
  y -= rhs.y; 
  return *this; 
};

__host__ __device__ vec2 vec2::operator* ( float rhs) const
{
  return vec2( x * rhs, y * rhs);
};

__host__ __device__ vec2 vec2::operator*= ( float rhs)
{
  x *= rhs;
  y *= rhs;
  return *this;
};

__device__ vec2::operator float() const 
{
  return sqrtf( x*x + y*y);
};

__host__ __device__ float vec2::lengthSquared() const
{
  return x*x + y*y;
};

__device__ vec2 vec2::normalize() {
  float lenght = (float) (*this);
  x /= lenght;
  y /= lenght;
  return *this;
};

__device__ vec2 vec2::normalized() const {
  float lenght = (float) (*this);
  return vec2( x / lenght, y / lenght);
};

__host__ __device__ float vec2::dot( const vec2& lhs, const vec2& rhs) {
  return lhs.x * rhs.x + lhs.y * rhs.y;
};

__host__ __device__ vec2 vec2::random() {
    thrust::default_random_engine randEng;
    thrust::uniform_real_distribution<float> uniDist(-(PI),(PI));
    return vec2( sinf(uniDist(randEng)), cosf(uniDist(randEng)));
};
