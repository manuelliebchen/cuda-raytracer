#include "scene.cuh"

Scene::Scene()
{

};

Scene::~Scene() {
};

bool
Scene::trace( Ray& ray, Intersection& intersection) const
{
  bool hit = false;

  for( int i = 0; i < hostSpheres.size(); i++) {
    hit |= hostSpheres[i].intersect(ray, intersection);
  }

  for( int i = 0; i < hostPlanes.size(); i++) {
    hit |= hostPlanes[i].intersect(ray, intersection);
  }

  return hit;
};

float3
Scene::phong( const Intersection& intersection) const {
    const Material& material = hostMaterials[intersection.materialIndex];
    float3 color = material.getBaseColor();

    if(material.getEmission() == 0) {
      float phong = 0.1;
      for( int i = 0; i < hostPlanes.size(); i++) {
        const Material& emMaterial = hostMaterials[hostPlanes[i].getMaterialIndex()];
        if( emMaterial.getEmission() > 0) {

          vec3 to_light = (hostPlanes[i].getEmissionPoint() - intersection.position);

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

void Scene::parseFile( JSONArray material, JSONArray objects) {
  for( JSONValue* i : material) {
    JSONObject v = i->AsObject();
    JSONArray color = v[(L"color")]->AsArray();
    hostMaterials.push_back(Material(make_float3(color[0]->AsNumber(), color[1]->AsNumber(), color[2]->AsNumber()),
                                 v[L"diffuse"]->AsNumber(),
                                 v[L"reflection"]->AsNumber(),
                                 v[L"refraction"]->AsNumber(),
                                 v[L"refractionIndex"]->AsNumber(),
                                 v[L"emission"]->AsNumber()));
  }
  hostMaterials.shrink_to_fit();

  for( JSONValue* i : objects) {
    JSONObject v = i->AsObject();
    const std::wstring type = v[L"type"]->AsString();
    vec3 position = vec3( v[L"position"]);
    unsigned material = v[L"material"]->AsNumber();
    if(type == L"plane") {
      JSONArray jsonXAchse = v[L"xAchse"]->AsArray();
      vec3 xAchse( v[L"xAchse"]);
      vec3 yAchse( v[L"yAchse"]);
      hostPlanes.push_back( Plane(position, material, xAchse, yAchse));
    } else if( type == L"sphere") {
      float radius = v[L"radius"]->AsNumber();
      hostSpheres.push_back( Sphere( position, material, radius));
    } else if( type == L"cube") {
      vec3 xAchse( v[L"xAchse"]);
      vec3 yAchse( v[L"yAchse"]);
      vec3 zAchse( v[L"zAchse"]);
      hostPlanes.push_back( Plane(position + zAchse, material, xAchse, yAchse));
      hostPlanes.push_back( Plane(position - zAchse, material, -yAchse, xAchse));
      hostPlanes.push_back( Plane(position + yAchse, material, xAchse, zAchse));
      hostPlanes.push_back( Plane(position - yAchse, material, -zAchse, xAchse));
      hostPlanes.push_back( Plane(position + xAchse, material, yAchse, zAchse));
      hostPlanes.push_back( Plane(position - xAchse, material, -zAchse, yAchse));
    }
  }
  hostSpheres.shrink_to_fit();
  hostPlanes.shrink_to_fit();
};
