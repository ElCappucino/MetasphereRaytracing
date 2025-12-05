#pragma once
#include "material.h"

#define LAMBERT 1
#define METAL 2
#define DIELECTRIC 3

#define EPS 1e-3
#define EPS_NORMAL 1e-3
#define ITR 128
#define BLENDING_CONST 2.0f

class metaspheres {
public:
    metaspheres()
    {
        int idx = 0;

        idx = 0;
        center[idx] = vec3(0, 1, 0);
        radius[idx] = 1.5;
        albedo[idx] = vec3(0.6, 0.6, 0.6);
        fuzz[idx] = 0.05f;
        ref_idx[idx] = 1.8f;
        mat_type[idx] = METAL;
        list_size = 1;

        idx = 1;
        center[idx] = vec3(4, 1, 0);
        radius[idx] = 0.5;
        albedo[idx] = vec3(0.6, 0.6, 0.6);
        fuzz[idx] = 0.05f;
        ref_idx[idx] = 1.8f;
        mat_type[idx] = METAL;
        list_size = 2;

        idx = 2;
        center[idx] = vec3(-4, 1, 0);
        radius[idx] = 0.5;
        albedo[idx] = vec3(0.6, 0.6, 0.6);
        fuzz[idx] = 0.05f;
        ref_idx[idx] = 1.8f;
        mat_type[idx] = METAL;
        list_size = 3;
    };

    __device__ bool hitall(const ray& r, float& t_min, float& t_max, hit_record& rec) const;
    __device__ bool hitMeta(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hitMetaSingle(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;

    __device__ static float sdSphere(vec3 origin, float radius);
    __device__ static float smoothMin(float d1, float d2, float k);
    __device__ float map(vec3 rayPos) const;
    __device__ vec3 generateNormal(vec3 rayPos) const;

    // this example use 20KB (max kernel const parameter 32KB)
    vec3 center[3];
    float radius[3];
    vec3 albedo[3];
    float fuzz[3];
    float ref_idx[3];
    int mat_type[3];
    int list_size;
};

__device__ float metaspheres::sdSphere(vec3 origin, float radius) {
    return origin.length() - radius;
}

__device__ float metaspheres::smoothMin(float d1, float d2, float k) {
    //float h = exp(-k * d1) + exp(-k * d2);
    //return -log(h) / k;

    // cubic polynomial
    float h = fmaxf(k - fabsf(d1 - d2), 0.0f) / k;
    return fminf(d1, d2) - h * h * h * k / 6.0f;
}

__device__ float metaspheres::map(vec3 rayPos) const {
    float distance = 1e5;
    float k = BLENDING_CONST;

    int sphereNum = list_size;
    for (int i = 0; i < sphereNum; i++) {
        float sphere = sdSphere(rayPos - center[i], radius[i]);
        distance = smoothMin(distance, sphere, k);
        //distance = sphere < distance ? sphere : distance; // normal min
    }

    return distance;
}

__device__ vec3 metaspheres::generateNormal(vec3 rayPos) const {
    return unit_vector(vec3(
        map(rayPos + vec3(EPS_NORMAL, 0.0, 0.0)) - map(rayPos + vec3(-EPS_NORMAL, 0.0, 0.0)),
        map(rayPos + vec3(0.0, EPS_NORMAL, 0.0)) - map(rayPos + vec3(0.0, -EPS_NORMAL, 0.0)),
        map(rayPos + vec3(0.0, 0.0, EPS_NORMAL)) - map(rayPos + vec3(0.0, 0.0, -EPS_NORMAL))
    ));
}

__device__ bool metaspheres::hitall(const ray& r, float& t_min, float& t_max, hit_record& rec) const 
{
    return hitMeta(r, t_min, t_max, rec);
}

__device__ bool metaspheres::hitMeta(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 rayPos = r.origin();
    vec3 dir = unit_vector(r.direction());
    float distance = 0.0f;
    float t = 0.0f;
    bool anyHit = false;
    for (int i = 0; i < ITR; i++) {
        distance = map(rayPos);

        vec3 v = rayPos - r.origin();
        float length = v.length();

        t += distance;
        rayPos = r.origin() + dir * t;

        if (distance < EPS && i > 0) {
            if (t >= t_min) {
                anyHit = true;
            }
        }

        //if (t > t_max) return false;
        if (anyHit) break;
    }

    if (!anyHit) return false;

    rec.t = t;
    //rec.p = r.point_at_parameter(t);
    rec.p = rayPos - dir * EPS * 2.0f;
    //rec.p = rayPos - dir * EPS * 1.0f;
    vec3 normal = generateNormal(rec.p);
    rec.normal = normal;

    //if (dot(rec.normal, dir) > 0.0f) {
    //    rec.normal = -rec.normal;
    //}

    int closestIdx = 0;
    float closestDistance = 0;

    for (int i = 0; i < list_size; i++) {
        float distance = sdSphere(rec.p - center[i], radius[i]);
        if (i == 0 || distance < closestDistance) {
            closestIdx = i;
            closestDistance = distance;
        }
    }

    rec.albedo = albedo[closestIdx];
    rec.fuzz = fuzz[closestIdx];
    rec.ref_idx = ref_idx[closestIdx];
    rec.mat_type = mat_type[closestIdx];

    return true;
}

__device__ bool metaspheres::hitMetaSingle(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 rayPos = r.origin();
    vec3 dir = unit_vector(r.direction());
    float distance = 0.0f;
    float t = 0.0f;
    bool anyHit = false;
    for (int i = 0; i < ITR; i++) {
        distance = distance = sdSphere(rayPos - center[idx], radius[idx]);

        vec3 v = rayPos - r.origin();
        float length = v.length();

        t += distance;
        rayPos = r.origin() + dir * t;

        if (distance < EPS && i > 0) {
            anyHit = true;
            break;
        }

        if (t > t_max) return false;
    }

    if (!anyHit) return false;
    if (t < t_min) return false;

    rec.t = t;
    rec.p = rayPos - dir * EPS * 2.0f;
    vec3 normal = generateNormal(rec.p);
    rec.normal = normal;


    rec.albedo = albedo[idx];
    rec.fuzz = fuzz[idx];
    rec.ref_idx = ref_idx[idx];
    rec.mat_type = mat_type[idx];

    return true;
}

__device__ bool metaspheres::hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center[idx];
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius[idx] * radius[idx];
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center[idx]) / radius[idx];
            rec.albedo = albedo[idx];
            rec.fuzz = fuzz[idx];
            rec.ref_idx = ref_idx[idx];
            rec.mat_type = mat_type[idx];
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center[idx]) / radius[idx];
            rec.albedo = albedo[idx];
            rec.fuzz = fuzz[idx];
            rec.ref_idx = ref_idx[idx];
            rec.mat_type = mat_type[idx];
            return true;
        }
    }
    return false;
}
