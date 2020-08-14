#include "deviceScene.cuh"

DeviceScene::DeviceScene( const Scene& _scene) :
  numSpheres( _scene.hostSpheres.size()),
  numPlanes( _scene.hostPlanes.size()),
  numMaterials( _scene.hostMaterials.size())
{
  totalMemory = numSpheres * sizeof(Sphere) + numPlanes * sizeof(Plane) + numMaterials * sizeof(Material);
  cudaMalloc( (void**) &memory, totalMemory);

  cudaMemcpy( memory, _scene.hostSpheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
  cudaMemcpy( memory + numSpheres * sizeof(Sphere), _scene.hostPlanes.data(), numPlanes * sizeof(Plane), cudaMemcpyHostToDevice);
  cudaMemcpy( memory + numSpheres * sizeof(Sphere) + numPlanes * sizeof(Plane), _scene.hostMaterials.data(), numMaterials * sizeof(Material), cudaMemcpyHostToDevice);
};

const DeviceScene&
DeviceScene::operator= ( const Scene& _scene)
{
  numSpheres = _scene.hostSpheres.size();
  numPlanes = _scene.hostPlanes.size();
  numMaterials = _scene.hostMaterials.size();

  cudaFree(memory);
  memory = nullptr;

  totalMemory = numSpheres * sizeof(Sphere) + numPlanes * sizeof(Plane) + numMaterials * sizeof(Material);
  cudaMalloc( (void**) &memory, totalMemory);

  cudaMemcpy( memory, _scene.hostSpheres.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
  cudaMemcpy( memory + numSpheres * sizeof(Sphere), _scene.hostPlanes.data(), numPlanes * sizeof(Plane), cudaMemcpyHostToDevice);
  cudaMemcpy( memory + numSpheres * sizeof(Sphere) + numPlanes * sizeof(Plane), _scene.hostMaterials.data(), numMaterials * sizeof(Material), cudaMemcpyHostToDevice);

  return (*this);
};

__host__ __device__
size_t
DeviceScene::getSharedMemory() const {
  return totalMemory;
};

DeviceScene::~DeviceScene() {
  cudaFree( memory);
};

__device__
void
DeviceScene::copyDataToShared( char sharedMemory[], unsigned int linearThreadId) {
  for(int i = linearThreadId; i < totalMemory; i += blockDim.x) {
    sharedMemory[i] = memory[i];
  }

  __syncthreads();

  deviceSpheres = (Sphere*) sharedMemory;
  devicePlanes = (Plane*)&deviceSpheres[numSpheres];
  deviceMaterials = (Material*)&devicePlanes[numPlanes];
};

__device__
void
DeviceScene::copyDataToShared( char sharedMemory[]) {
  for(int i = 0; i < totalMemory; ++i) {
    sharedMemory[i] = memory[i];
  }

  __syncthreads();

  deviceSpheres = (Sphere*) sharedMemory;
  devicePlanes = (Plane*)&deviceSpheres[numSpheres];
  deviceMaterials = (Material*)&devicePlanes[numPlanes];
};

__device__
bool
DeviceScene::trace( Ray& ray, Intersection& intersection) const
{
  bool hit = false;

  for( int i = 0; i < numSpheres; i++) {
    Sphere sphere = deviceSpheres[i];
    hit |= sphere.intersect(ray, intersection);
  }

  for( int i = 0; i < numPlanes; i++) {
    Plane plane = devicePlanes[i];
    hit |= plane.intersect(ray, intersection);
  }

  return hit;
};

__device__
float3
DeviceScene::phong( const Intersection& intersection) const {
    Material material = deviceMaterials[intersection.materialIndex];
    float3 color = material.getBaseColor();

    if(material.getEmission() == 0) {
      float phong = 0.1;
      for( int i = 0; i < numPlanes; i++) {
        Plane plane = devicePlanes[i];
        Material emMaterial = deviceMaterials[plane.getMaterialIndex()];
        if( emMaterial.getEmission() > 0) {

          vec3 to_light = (plane.getEmissionPoint() - intersection.position);

          Intersection shadow_intersection;
          Ray ray( intersection.position + to_light.normalized() * EPSILON, to_light.normalized(), ((float) to_light) - EPSILON * 2);
          if( trace(ray,shadow_intersection)){
            continue;
          }

          float kd = vec3::dot( to_light.normalized(), intersection.normal) * material.getDiffuse();
          kd = max(0.0f, kd);
          float ks = vec3::dot( to_light.normalized(), intersection.reflection);
          if(ks > 0) {
            ks = pow(ks,32) * material.getSpecular();
          } else {
            ks = 0;
          }
          phong += kd + ks;
        }
      }
      phong = max(0.0f, phong);
      color.x *= phong;
      color.y *= phong;
      color.z *= phong;
    }
    return color;
}
