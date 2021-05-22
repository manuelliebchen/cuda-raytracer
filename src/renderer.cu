#include "renderer.cuh"
namespace crt {

Renderer::Renderer()
  : multisamples(1){};

Renderer::~Renderer(){};

Renderer::Renderer(const Renderer& _renderer)
  : camera(_renderer.camera)
  , image(_renderer.image)
  , scene(_renderer.scene)
  , multisamples(_renderer.multisamples){};

const Renderer
Renderer::operator=(const Renderer& _renderer)
{
  camera = _renderer.camera;
  image = _renderer.image;
  scene = _renderer.scene;
  multisamples = _renderer.multisamples;
  return (*this);
};

void
Renderer::readeSceneFile(std::string file)
{

  std::ifstream t(file);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  t.close();

  JSONValue* value = JSON::Parse(str.c_str());

  if (value) {
    JSONObject document = value->AsObject();
    JSONObject cameraDoc = document[L"camera"]->AsObject();
    vec3 cameraPosition = vec3(cameraDoc[L"position"]);
    vec3 cameraDirection = vec3(cameraDoc[L"direction"]);
    cameraDirection = cameraDirection.normalize() *
                      (float)cameraDoc[L"focallength"]->AsNumber();
    camera = Camera(cameraPosition, cameraDirection);

    JSONObject json_image = document[L"image"]->AsObject();
    image = Image(json_image[L"width"]->AsNumber(),
                  json_image[L"height"]->AsNumber());

    JSONObject settings = document[L"settings"]->AsObject();
    multisamples = settings[L"multisamples"]->AsNumber();

    scene.parseFile(document[L"material"]->AsArray(),
                    document[L"scene"]->AsArray());
  } else {
    std::cerr << "File: " << file << " not Parseable!" << std::endl;
    exit(1);
  }

  delete value;
};

void
Renderer::runOnHost()
{

  int multisamples_pattern[42] = { 0, 0,  -2, -6, 6,  -2, -6, 2,  2,  6,  1,
                                   1, -1, -3, -3, 2,  4,  -1, -5, -2, 2,  5,
                                   5, 3,  3,  -5, -2, 6,  0,  7,  -4, -6, -6,
                                   4, -8, 0,  7,  -4, 6,  7,  -7, -8 };
  unsigned int pattern_start = 0;
  if (multisamples == 4) {
    pattern_start = 2;
  } else if (multisamples == 16) {
    pattern_start = 10;
  }
  for (unsigned int i = 0; i < image.getHeight(); ++i) {
    for (unsigned int j = 0; j < image.getWidth(); ++j) {
      float3 floatColorSummarized = make_float3(0, 0, 0);
      for (unsigned int k = 0; k < multisamples; ++k) {
        uint2 imagePosition = make_uint2(j, i);
        float pixel_angle = (float)(image.getWidth() + image.getHeight()) / 2;
        Ray ray = camera.getRay(make_float2(
          ((imagePosition.x) - (0.5f * image.getWidth()) +
           ((float)multisamples_pattern[pattern_start + k * 2]) / 16.0f) /
            (pixel_angle),
          ((imagePosition.y) - (0.5f * image.getHeight()) +
           ((float)multisamples_pattern[pattern_start + k * 2 + 1]) / 16.0f) /
            (pixel_angle)));

        Intersection intersection;
        float3 floatColor = make_float3(0, 0, 0);
        if (scene.trace(ray, intersection)) {
          floatColor = scene.phong(intersection);
        }
        floatColorSummarized.x += floatColor.x;
        floatColorSummarized.y += floatColor.y;
        floatColorSummarized.z += floatColor.z;
      }
      floatColorSummarized.x /= multisamples;
      floatColorSummarized.y /= multisamples;
      floatColorSummarized.z /= multisamples;

      uchar3 color =
        make_uchar3((unsigned char)(floatColorSummarized.x * 255.0f),
                    (unsigned char)(floatColorSummarized.y * 255.0f),
                    (unsigned char)(floatColorSummarized.z * 255.0f));
      image.setPixel(make_uint2(j, i), color);
    }
  }
}

void
Renderer::runOnDevice()
{
  int deviceCout = 0;
  cudaGetDeviceCount(&deviceCout);
  cudaCheckError();
  assert(deviceCout > 0);

  int deviceHandle = 0;
  cudaSetDevice(deviceHandle);

  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, deviceHandle);

  DeviceRenderer* lokalDeviceRenderer = new DeviceRenderer(*this);
  DeviceRenderer* deviceRenderer;
  cudaMalloc((void**)&deviceRenderer, sizeof(DeviceRenderer));

  cudaMemcpy((void*)deviceRenderer,
             (void*)lokalDeviceRenderer,
             sizeof(DeviceRenderer),
             cudaMemcpyHostToDevice);

  cudaCheckError();

  unsigned int threadDimension =
    floorf(sqrt(deviceProperties.maxThreadsPerBlock));
  dim3 threadsPerBlock(threadDimension, threadDimension, 1);

  dim3 numberOfBlocks(
    (unsigned int)std::ceil((float)(multisamples * image.getWidth()) /
                            (float)(threadsPerBlock.x)),
    (unsigned int)std::ceil((float)(multisamples * image.getHeight()) /
                            (float)(threadsPerBlock.y)),
    1);

  auto start_device = std::chrono::high_resolution_clock::now();
  globalRender<<<numberOfBlocks,
                 threadsPerBlock,
                 lokalDeviceRenderer->getSharedMemory()>>>(deviceRenderer);
  cudaDeviceSynchronize();

  auto end_device = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_device = end_device - start_device;
  std::cout << "Render took: " << time_device.count() << "s" << std::endl;

  cudaCheckError();

  cudaMemcpy((void*)lokalDeviceRenderer,
             (void*)deviceRenderer,
             sizeof(DeviceRenderer),
             cudaMemcpyDeviceToHost);

  image = lokalDeviceRenderer->image;

  delete lokalDeviceRenderer;
  cudaFree(deviceRenderer);
};

void
Renderer::runOnDeviceSingleThreadet()
{
  int deviceCout = 0;
  cudaGetDeviceCount(&deviceCout);
  cudaCheckError();
  assert(deviceCout > 0);

  int deviceHandle = 0;
  cudaSetDevice(deviceHandle);

  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, deviceHandle);

  DeviceRenderer* lokalDeviceRenderer = new DeviceRenderer(*this);
  DeviceRenderer* deviceRenderer;
  cudaMalloc((void**)&deviceRenderer, sizeof(DeviceRenderer));

  cudaMemcpy((void*)deviceRenderer,
             (void*)lokalDeviceRenderer,
             sizeof(DeviceRenderer),
             cudaMemcpyHostToDevice);

  cudaCheckError();

  auto start_device = std::chrono::high_resolution_clock::now();
  globalRenderSingleThreadet<<<1, 1, lokalDeviceRenderer->getSharedMemory()>>>(
    deviceRenderer);
  cudaDeviceSynchronize();
  cudaCheckError();

  auto end_device = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_device = end_device - start_device;
  std::cout << "Render took: " << time_device.count() << "s" << std::endl;

  cudaCheckError();

  cudaMemcpy((void*)lokalDeviceRenderer,
             (void*)deviceRenderer,
             sizeof(DeviceRenderer),
             cudaMemcpyDeviceToHost);

  image = lokalDeviceRenderer->image;

  delete lokalDeviceRenderer;
  cudaFree(deviceRenderer);
};

void
Renderer::writeImage(std::string imageFile) const
{
  image.write(imageFile);
};

__global__ void
globalRenderSingleThreadet(DeviceRenderer* renderer)
{
  renderer->renderSingleThreadet();
};

__global__ void
globalRender(DeviceRenderer* renderer)
{
  renderer->render();
};

};
