#include "image.cuh"

Image::Image( int _width, int _height) :
  width(_width),
  height(_height),
  memory(_width * _height * 3)
{
  hostData = new unsigned char [ memory];
};

Image::Image() :
  width(0),
  height(0),
  memory(0),
  hostData( nullptr)
{
};

Image::Image( const Image& _image) :
  width( _image.width),
  height( _image.height),
  memory( _image.memory)
{
  hostData = new unsigned char [ memory];
  std::copy( _image.hostData, _image.hostData + memory, hostData);
};

Image Image::operator= (const Image& _image) {
  width = _image.width;
  height = _image.height;
  memory = _image.memory;
  delete[] hostData;
  hostData = new unsigned char [ memory];
  std::copy( _image.hostData, _image.hostData + memory, hostData);
  return (*this);
};

Image Image::operator= (const DeviceImage& _image) {
  width = _image.width;
  height = _image.height;
  memory = _image.memory;
  delete[] hostData;
  hostData = new unsigned char [ memory];
  cudaMemcpy( (void*) hostData, (void*) _image.deviceData, memory * sizeof( unsigned char), cudaMemcpyDeviceToHost);
  return (*this);
};

Image::~Image() {
  delete[] hostData;
};

unsigned Image::getWidth() {
  return width;
};

unsigned Image::getHeight() {
  return height;
};

void Image::setPixel( uint2 position, uchar3 value) {
  if(position.x < width && position.y < height){
    hostData[ (position.y * width + position.x) * 3 + 0] = value.x;
    hostData[ (position.y * width + position.x) * 3 + 1] = value.y;
    hostData[ (position.y * width + position.x) * 3 + 2] = value.z;
  }
};

void Image::write( std::string imageFile ) const {
  FILE *fp = fopen( imageFile.c_str(), "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);

  std::string description("Copyright 2018 Manuel Liebchen\nfor GPU-Programming at OVGU-Magdeburg\nby Christian Lessig");
  png_text text_ptr[1];
  text_ptr[0].key = const_cast<char*>("Description");
  text_ptr[0].text = const_cast<char*>(description.c_str());
  text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE;
  png_set_text (png_ptr, info_ptr, text_ptr, 1);

  /* write header */
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr, width, height,
      8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);


  png_bytep * row_pointers = new png_bytep [height];
  int j = 0;
  for(int i = height-1; i >= 0; --i){
    row_pointers[j++] = hostData + (i * width * 3);
  }
  //allocate row_pointers and store each row of your image in it
  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, NULL);

  delete[] row_pointers;
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
};
