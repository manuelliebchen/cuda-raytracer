#include "deviceImage.cuh"

namespace crt {
DeviceImage::DeviceImage()
  : width(0)
  , height(0)
  , memory(0)
{
  deviceData = nullptr;
};

DeviceImage::DeviceImage(int _width, int _height)
  : width(_width)
  , height(_height)
  , memory(_width * _height * 3)
{
  cudaMalloc((void**)&deviceData, memory * sizeof(unsigned char));
};

DeviceImage::DeviceImage(const Image& _image)
  : width(_image.width)
  , height(_image.height)
  , memory(_image.memory)
{
  cudaMalloc((void**)&deviceData, memory * sizeof(unsigned char));

  cudaCheckError();
};

DeviceImage
DeviceImage::operator=(const Image& _image)
{
  width = _image.width;
  height = _image.height;
  memory = _image.memory;

  cudaMalloc((void**)&deviceData, memory * sizeof(unsigned char));
  cudaMemcpy((void*)deviceData,
             _image.hostData,
             memory * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  return (*this);
};

size_t
DeviceImage::getMemoryUsage() const
{
  return memory;
}

DeviceImage::~DeviceImage()
{
  cudaFree(deviceData);
};

__device__ unsigned
DeviceImage::getWidth() const
{
  return width;
};

__device__ unsigned
DeviceImage::getHeight() const
{
  return height;
};

__device__ void
DeviceImage::setPixel(uint2 position, uchar3 value)
{
  if (position.x < width && position.y < height) {
    deviceData[(position.y * width + position.x) * 3 + 0] = value.x;
    deviceData[(position.y * width + position.x) * 3 + 1] = value.y;
    deviceData[(position.y * width + position.x) * 3 + 2] = value.z;
  }
};
};