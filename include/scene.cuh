#ifndef _SCENE_
#define _SCENE_

#include <vector>
#include <stdio.h>

//For Parsing
#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>

#include <algorithm>

#include "util.cuh"

#include "deviceScene.cuh"

#include "material.cuh"

#include "object.cuh"
#include "sphere.cuh"
#include "plane.cuh"

#include "JSON.h"
#include "JSONValue.h"

class Scene {
  friend class DeviceScene;

  public:
    Scene();
    ~Scene();

    void parseFile( JSONArray material, JSONArray objects);
    bool trace( Ray& ray, Intersection& intersection) const;
    float3 phong( const Intersection& intersection) const;

  private:

    std::vector<Sphere> hostSpheres;

    std::vector<Plane> hostPlanes;

    std::vector<Material> hostMaterials;
};

#endif
