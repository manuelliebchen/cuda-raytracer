#include "main.cuh"
#include <chrono>

int main (int argc, char ** argv) {
  Renderer renderer;
  std::cout << "Reading Scene File: " << argv[1] << std::endl;
  renderer.readeSceneFile( argv[1]);

#if 0 // DEVICE SINGLE THREAD
  std::cout << "Running on Device, Single Thread." << std::endl;
  auto start_single = std::chrono::high_resolution_clock::now();

  renderer.runOnDeviceSingleThreadet();

  auto end_single = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_single = end_single - start_single;
  std::cout << "With Copy it took: " << time_single.count() << "s" << std::endl;

  std::cout << "Writing Image to " << argv[2] << "device_render_single.png" << std::endl;
  renderer.writeImage( std::string(argv[2]) + "device_render_single.png" );
#endif

#if 1 // DEVICE
  std::cout << "Running on Device." << std::endl;
  auto start_device = std::chrono::high_resolution_clock::now();

  renderer.runOnDevice();

  auto end_device = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_device = end_device - start_device;
  std::cout << "With Copy it took: " << time_device.count() << "s" << std::endl;

  std::cout << "Writing Image to " << argv[2] << "device_render.png" << std::endl;
  renderer.writeImage( std::string(argv[2]) + "device_render.png" );
#endif

#if 0 // HOST
  std::cout << "Running on Host." << std::endl;
  auto start_host = std::chrono::high_resolution_clock::now();

  renderer.runOnHost();

  auto end_host = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_host = end_host - start_host;
  std::cout << "Took: " << time_host.count() << "s" << std::endl;

  std::cout << "Writing Image to " << argv[2] << "host_render.png" << std::endl;
  renderer.writeImage( std::string(argv[2]) + "host_render.png" );
#endif

};
