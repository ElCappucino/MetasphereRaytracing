#ifndef HITABLELISTH
#define HITABLELISTH

#include "metasphere.h"
#include "ray.h"


class hitable_list {
    public:
        hitable_list() {}
        __device__ bool hitall(const ray& r, float tmin, float tmax, hit_record& rec) const;
        metaspheres meta_world;
};

__device__ bool hitable_list::hitall(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        hit_record closest_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        if (meta_world.hitall(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            closest_rec = temp_rec;
        }
        
        if (hit_anything) {
            rec = closest_rec;
        }

        return hit_anything;
}

#endif

