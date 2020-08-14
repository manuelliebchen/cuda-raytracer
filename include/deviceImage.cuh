#ifndef _DEVICE_IMAGE_
#define _DEVICE_IMAGE_

#include <stdio.h>
#include <string>
#include <iostream>

#include "image.cuh"
#include "util.cuh"

class Image;

class DeviceImage {
  friend class Image;

  public:
    DeviceImage();
    DeviceImage( int _width, int _height);
    DeviceImage( const Image& _image);
    DeviceImage operator = (const Image& _image);
    ~DeviceImage();

    __host__ __device__ unsigned int getWidth() const;
    __host__ __device__ unsigned int getHeight() const;

    __device__ void setPixel( uint2 position, uchar3 value);

    size_t getMemoryUsage() const;

  private:
    unsigned int width;
    unsigned int height;
    size_t memory;

    unsigned char * deviceData;
};

#endif //_DEVICE_IMAGE_
