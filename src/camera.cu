#include "camera.cuh"

namespace crt {
Camera::Camera(vec3 _position, vec3 _direction)
  : position(_position)
  , direction(_direction){};

Camera::Camera()
  : position(vec3(0, 0, 0))
  , direction(vec3(1, 0, 0)){};

Camera::Camera(const Camera& _camera)
  : position(_camera.position)
  , direction(_camera.direction){};

Camera
Camera::operator=(const Camera& _camera)
{
  position = _camera.position;
  direction = _camera.direction;
  return *this;
};

Camera::~Camera(){

};

__device__ Ray
Camera::getRay(float2 pixelposition)
{
  return Ray(position, direction + vec3(pixelposition.x, pixelposition.y, 0));
};
};