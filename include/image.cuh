#ifndef IMAGE
#define IMAGE

#include <stdio.h>
#include <string>

#include <iostream>
#include <algorithm>

#include "deviceImage.cuh"

#define PNG_DEBUG 3
#include <png.h>

class DeviceImage;

class Image {

  friend class DeviceImage;

  public:
    Image( int _width, int _height);
    Image( );
    Image( const Image& _image);
    Image operator= (const Image& _image);
    Image operator= (const DeviceImage& _image);
    ~Image();

    void setPixel( uint2 position, uchar3 value);

    unsigned int getWidth();
    unsigned int getHeight();

    void write( std::string imageFile ) const;

  private:
    unsigned int width;
    unsigned int height;
    size_t memory;

    unsigned char * hostData;
};

#endif
