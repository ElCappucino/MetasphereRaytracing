#include <iostream>
#include <chrono>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"

#define MAXFLOAT FLT_MAX
#define M_PI 3.1415926

vec3 color(const ray& r, hitable *world, int depth) {
    hit_record rec;
    if (world->hit(r, 0.001, MAXFLOAT, rec)) {
        ray scattered;
        vec3 attenuation;
        if (depth < 10 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
             return attenuation*color(scattered, world, depth+1);
        }
        else {
            return vec3(0,0,0);
        }
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}


hitable *random_scene() {
    int n = 500;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = (rand() / (RAND_MAX + 1.0));
            vec3 center(a+0.9* (rand() / (RAND_MAX + 1.0)),0.2,b+0.9* (rand() / (RAND_MAX + 1.0)));
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new lambertian(vec3((rand() / (RAND_MAX + 1.0)) * (rand() / (RAND_MAX + 1.0)), (rand() / (RAND_MAX + 1.0)) * (rand() / (RAND_MAX + 1.0)), (rand() / (RAND_MAX + 1.0)) * (rand() / (RAND_MAX + 1.0)))));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + (rand() / (RAND_MAX + 1.0))), 0.5*(1 + (rand() / (RAND_MAX + 1.0))), 0.5*(1 + (rand() / (RAND_MAX + 1.0)))),  0.5* (rand() / (RAND_MAX + 1.0))));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    return new hitable_list(list,i);
}

int main() {
    int nx = 1800;
    int ny = 1000;
    int ns = 16;

  
    vec3* fb;
    fb = (vec3*)malloc(nx * ny * sizeof(vec3));
    int idx = 0;

    hitable *list[5];
    float R = cos(M_PI/4);
    list[0] = new sphere(vec3(0,0,-1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
    list[1] = new sphere(vec3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
    list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
    list[3] = new sphere(vec3(-1,0,-1), 0.5, new dielectric(1.5));
    list[4] = new sphere(vec3(-1,0,-1), -0.45, new dielectric(1.5));
    hitable *world = new hitable_list(list,5);
    world = random_scene();

    vec3 lookfrom(0,2,20);
    vec3 lookat(0,2,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(0, 0, 0);
            for (int s=0; s < ns; s++) {
                float u = float(i + (rand() / (RAND_MAX + 1.0))) / float(nx);
                float v = float(j + (rand() / (RAND_MAX + 1.0))) / float(ny);
                ray r = cam.get_ray(u, v);
                vec3 p = r.point_at_parameter(2.0);
                col += color(r, world,0);
            }
            col /= float(ns);
            col = vec3(255.99 * sqrt(col[0]), 255.99 * sqrt(col[1]), 255.99 * sqrt(col[2]) );
            //int ir = int(255.99*col[0]);
            //int ig = int(255.99*col[1]);
            //int ib = int(255.99*col[2]);
            fb[idx++] = col;
            //fprintf(fp, "%i %i %i\n", ir, ig, ib);
            //std::cout << ir << " " << ig << " " << ib << "\n";
        }
        //std::cout << j << "\n";
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "render time " << time_span.count() << " seconds. " << std::endl;

    // output to ppm
    FILE* fp;
    fp = fopen("img.ppm", "w");
    fprintf(fp, "P3\n %i %i \n255\n", nx, ny);
    //std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    idx = 0;
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            fprintf(fp, "%i %i %i\n", (int)fb[idx].x(), (int)fb[idx].y(), (int)fb[idx].z());
            idx++;
        }
    }
    fclose(fp);


    free(fb);

}
