#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera {
public:
    camera(){}
    //camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
    //    lens_radius = aperture / 2.0f;
    //    float theta = vfov*((float)M_PI)/180.0f;
    //    float half_height = tan(theta/2.0f);
    //    float half_width = aspect * half_height;
    //    origin = lookfrom;
    //    w = unit_vector(lookfrom - lookat);
    //    u = unit_vector(cross(vup, w));
    //    v = cross(w, u);
    //    lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
    //    horizontal = 2.0f*half_width*focus_dist*u;
    //    vertical = 2.0f*half_height*focus_dist*v;
    //}

    void updateCam(int dir, float _yaw, float _pitch, vec3 &lookfrom, vec3 &lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        float yaw = _yaw;
        float pitch = _pitch;

        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        front = glm::normalize(front)*0.5f;
        glm::vec3 right;
        right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0))) * 0.5f;

        glm::vec3 up = glm::normalize(glm::cross(right, front)) * 0.5f;
        
        switch (dir) {
            case 1:
                lookfrom.e[0] = lookfrom.e[0] + front.x;
                lookfrom.e[1] = lookfrom.e[1] + front.y;
                lookfrom.e[2] = lookfrom.e[2] + front.z;
                break;
            case 2:
                lookfrom.e[0] = lookfrom.e[0] - front.x;
                lookfrom.e[1] = lookfrom.e[1] - front.y;
                lookfrom.e[2] = lookfrom.e[2] - front.z;
                break;
            case 4:
                lookfrom.e[0] = lookfrom.e[0] + right.x;
                lookfrom.e[1] = lookfrom.e[1] + right.y;
                lookfrom.e[2] = lookfrom.e[2] + right.z;
                break;
            case 3:
                lookfrom.e[0] = lookfrom.e[0] - right.x;
                lookfrom.e[1] = lookfrom.e[1] - right.y;
                lookfrom.e[2] = lookfrom.e[2] - right.z;
                break;
            case 5:
                lookfrom.e[0] = lookfrom.e[0] + up.x;
                lookfrom.e[1] = lookfrom.e[1] + up.y;
                lookfrom.e[2] = lookfrom.e[2] + up.z;
                break;
            case 6:
                lookfrom.e[0] = lookfrom.e[0] - up.x;
                lookfrom.e[1] = lookfrom.e[1] - up.y;
                lookfrom.e[2] = lookfrom.e[2] - up.z;
                break;
            default:
                break;
        }

        lookat.e[0] = lookfrom.e[0] + front.x;
        lookat.e[1] = lookfrom.e[1] + front.y;
        lookat.e[2] = lookfrom.e[2] + front.z;

        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    // will pass this camera class to kernel as const para, so we have to define const member f()
    __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u* rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }


    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif
