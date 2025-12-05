#ifndef SPHEREH
#define SPHEREH


#include "material.h"



float rand01() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

class spheres {
public:
    spheres()
    {
        int idx = 0;
        center[idx] = vec3(0, -1000.0, -1);
        radius[idx] = 1000;
        albedo[idx] = vec3(0.5, 0.5, 0.5);
        mat_type[idx] = 1;
        idx = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = rand01();
                vec3 _center(a + rand01(), 0.2, b + rand01());
                if (choose_mat < 0.8f) {
                    center[idx] = _center;
                    radius[idx] = 0.2;
                    albedo[idx] = vec3(rand01()* rand01(), rand01() * rand01(), rand01() * rand01());
                    mat_type[idx] = 1;
                    idx++;
                }
                else if (choose_mat < 0.9f) {
                    center[idx] = _center;
                    radius[idx] = 0.2;
                    albedo[idx] = vec3(0.5f * (1.0f + rand01()), 0.5f * (1.0f + rand01()), 0.5f * (1.0f + rand01()));
                    fuzz[idx] = 0.5f * rand01();
                    mat_type[idx] = 2;
                    idx++;
                }
                else {
                    center[idx] = _center;
                    radius[idx] = 0.2;
                    ref_idx[idx] = 1.5;
                    mat_type[idx] = 3;
                    idx++;
                }
            }
        }
        center[idx] = vec3(0, 1,0);
        radius[idx] = 1;
        ref_idx[idx] = 1.5;
        mat_type[idx] = 3;
        idx++;
        center[idx] = vec3(-4, 1, 0);
        radius[idx] = 1;
        albedo[idx] = vec3(0.4, 0.2, 0.1);
        mat_type[idx] = 1;
        idx++;
        center[idx] = vec3(4, 1, 0);
        radius[idx] = 1;
        albedo[idx] = vec3(0.7, 0.6, 0.5);
        fuzz[idx] = 0.0;
        mat_type[idx] = 2;
        idx++;
        list_size = 22 * 22 + 1 + 3;
    };

    __device__ bool hitall(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool hit(int idx, const ray& r, float tmin, float tmax, hit_record& rec) const;


    // this example use 20KB (max kernel const parameter 32KB)
    vec3 center[500];
    float radius[500];
    vec3 albedo[500];
    float fuzz[500];
    float ref_idx[500];
    int mat_type[500];
    int list_size;
};

__device__ bool spheres::hitall(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (hit(i, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool spheres::hit(int idx, const ray& r, float t_min, float t_max, hit_record& rec) const {
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



//class sphere : public hitable {
//public:
//    __device__ sphere() {}
//    __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m) {};
//    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
//    vec3 center;
//    float radius;
//    material* mat_ptr;
//};
//
//__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//    vec3 oc = r.origin() - center;
//    float a = dot(r.direction(), r.direction());
//    float b = dot(oc, r.direction());
//    float c = dot(oc, oc) - radius * radius;
//    float discriminant = b * b - a * c;
//    if (discriminant > 0) {
//        float temp = (-b - sqrt(discriminant)) / a;
//        if (temp < t_max && temp > t_min) {
//            rec.t = temp;
//            rec.p = r.point_at_parameter(rec.t);
//            rec.normal = (rec.p - center) / radius;
//            rec.mat_ptr = mat_ptr;
//            return true;
//        }
//        temp = (-b + sqrt(discriminant)) / a;
//        if (temp < t_max && temp > t_min) {
//            rec.t = temp;
//            rec.p = r.point_at_parameter(rec.t);
//            rec.normal = (rec.p - center) / radius;
//            rec.mat_ptr = mat_ptr;
//            return true;
//        }
//    }
//    return false;
//}

#endif
