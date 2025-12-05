#include <iostream>
#include <time.h>
#include <float.h>


#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


#define STB_IMAGE_IMPLEMENTATION
#include "includes/stb_image.h"
#include "includes/KHR/khrplatform.h"
#include "includes/glew.h"
#include "includes/GLFW/glfw3.h"
#include "includes/GLFW/glfw3native.h"
#include "includes/shader.h"

// must include after gl lib
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include <windows.h>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <float.h>


#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
#define WIDTH 800
#define HEIGHT 600
// cuda rt
int nx = WIDTH;
int ny = HEIGHT;
int ns = 16;
int tx = 24;
int ty = 24;
int num_pixels = WIDTH * HEIGHT;
// cam
float Yaw = -90;
float Pitch = 0;
vec3 lookfrom(0, 2, 20);
vec3 lookat(0, 2, 0);
float dist_to_focus = 10.0;
float aperture = 0.001;
camera d_camera;

hitable_list* world_ptr;

double prevTime = 0.0;
double currTime = 0.0;
double dt = 0.0;

// Windows
HWND handle;
WNDPROC currentWndProc;
MSG Msg;
WNDPROC btnWndProc;
std::stringstream ss;

// Touch
#define MAXPOINTS 20
int points[MAXPOINTS][2];       // touch coor
int diff_points[MAXPOINTS][2];  // touch offset each frame
int idLookup[MAXPOINTS];
int last_points[MAXPOINTS][2];

// mouse
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool mouse1Pressed = true;

int currentIdx = -1;

// cuda opengl interop
GLuint shDrawTex;           // shader
GLuint tex_cudaResult;      // result texture to glBindTexture(GL_TEXTURE_2D, texture);
unsigned int* cuda_dest_resource;  // output from cuda
struct cudaGraphicsResource* cuda_tex_result_resource;

// orbit sphere
float orbitTime = 0.0f;

// ---------------------------------------
// Touch handler
// ---------------------------------------

// This function is used to return an index given an ID
int GetContactIndex(int dwID) {
    for (int i = 0; i < MAXPOINTS; i++) {
        if (idLookup[i] == dwID) {
            return i;
        }
    }

    for (int i = 0; i < MAXPOINTS; i++) {
        if (idLookup[i] == -1) {
            idLookup[i] = dwID;
            return i;
        }
    }
    // Out of contacts
    return -1;
}

// Mark the specified index as initialized for new use
BOOL RemoveContactIndex(int index) {
    if (index >= 0 && index < MAXPOINTS) {
        idLookup[index] = -1;
        return true;
    }

    return false;
}

LRESULT OnTouch(HWND hWnd, WPARAM wParam, LPARAM lParam) {
    BOOL bHandled = FALSE;
    UINT cInputs = LOWORD(wParam);
    PTOUCHINPUT pInputs = new TOUCHINPUT[cInputs];
    POINT ptInput;
    if (pInputs) {
        if (GetTouchInputInfo((HTOUCHINPUT)lParam, cInputs, pInputs, sizeof(TOUCHINPUT))) {
            for (UINT i = 0; i < cInputs; i++) {
                TOUCHINPUT ti = pInputs[i];
                int index = GetContactIndex(ti.dwID);
                if (ti.dwID != 0 && index < MAXPOINTS) {

                    // Do something with your touch input handle
                    ptInput.x = TOUCH_COORD_TO_PIXEL(ti.x);
                    ptInput.y = TOUCH_COORD_TO_PIXEL(ti.y);
                    ScreenToClient(hWnd, &ptInput);

                    if (ti.dwFlags & TOUCHEVENTF_UP) {
                        points[index][0] = -1;
                        points[index][1] = -1;
                        last_points[index][0] = -1;
                        last_points[index][1] = -1;
                        diff_points[index][0] = 0;
                        diff_points[index][1] = 0;

                        // Remove the old contact index to make it available for the new incremented dwID.
                        // On some touch devices, the dwID value is continuously incremented.
                        RemoveContactIndex(index);
                    }
                    else {
                        if (points[index][0] > 0) {
                            last_points[index][0] = points[index][0];
                            last_points[index][1] = points[index][1];
                        }

                        points[index][0] = ptInput.x;
                        points[index][1] = ptInput.y;

                        if (last_points[index][0] > 0) {
                            diff_points[index][0] = points[index][0] - last_points[index][0];
                            diff_points[index][1] = points[index][1] - last_points[index][1];
                        }
                    }
                }
            }
            bHandled = TRUE;
        }
        else {
            /* handle the error here */
        }
        delete[] pInputs;
    }
    else {
        /* handle the error here, probably out of memory */
    }
    if (bHandled) {
        // if you handled the message, close the touch input handle and return
        CloseTouchInputHandle((HTOUCHINPUT)lParam);
        return 0;
    }
    else {
        // if you didn't handle the message, let DefWindowProc handle it
        return DefWindowProc(hWnd, WM_TOUCH, wParam, lParam);
    }
}

LRESULT CALLBACK SubclassWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_TOUCH:
        OnTouch(hWnd, wParam, lParam);
        break;
    case WM_LBUTTONDOWN:
    {

    }
    break;
    case WM_CLOSE:
        DestroyWindow(hWnd);
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }

    return CallWindowProc(btnWndProc, hWnd, uMsg, wParam, lParam);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        d_camera.updateCam(1, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        d_camera.updateCam(2, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        d_camera.updateCam(3, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        d_camera.updateCam(4, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        d_camera.updateCam(5, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        d_camera.updateCam(6, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

bool pixelRayHit(vec3 center, float radius, const ray& r, float t_min, float t_max, float& minDist) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        minDist = temp;

        if (temp < t_max && temp > t_min) return true;
        temp = (-b + sqrt(discriminant)) / a;
        minDist = temp;

        if (temp < t_max && temp > t_min) return true;
    }
    return false;
}

ray getScreenRay(float s, float t, const camera& cam) {
    vec3 offset = cam.u + cam.v;
    vec3 A = cam.origin;
    vec3 B = cam.lower_left_corner + s * cam.horizontal + t * cam.vertical - cam.origin;
    ray r;
    r.A = A;
    r.B = unit_vector(B);
    return r;
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    if (world_ptr == nullptr) return;

    mouse1Pressed = false;
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        mouse1Pressed = true;
    }

    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);
    
    printf("current idx: %d\n", currentIdx);

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    xoffset *= 2.0f * dt;
    yoffset *= 2.0f * dt;

    float u = float(xpos) / float(nx);
    float v = 1.0f - (float(ypos) / float(ny));

    lastX = xpos;
    lastY = ypos;


    // Move camera angle by holding right click
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) 
    {
        Yaw += xoffset;
        Pitch += yoffset;

        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;

        d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);
    }

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) 
{
    
}


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        // Make sure we call CUDA Device Reset before exiting
        ss << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");
        cudaDeviceReset();
        exit(99);
    }
}

__device__ int clamp(int x, int a, int b) { return MAX(a, MIN(b, x)); }

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b) {
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);

    return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ vec3 sample_env(const vec3& dir, const float* envMap, int width, int height) {
    float u = (atan2f(dir.z(), dir.x()) + M_PI) / (2.0f * M_PI);
    float v = acosf(dir.y()) / M_PI;

    int x = (int)(u * width);
    int y = (int)(v * height);

    int idx = (y * width + x) * 3;

    return vec3(envMap[idx], envMap[idx + 1], envMap[idx + 2]);
}


__device__ vec3 color(const ray& r, const hitable_list world, curandState* local_rand_state, const float* envHDR, int hdrWidth, int hdrHeight) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 10; i++) {
        hit_record rec;
        float t_min = EPS;
        float t_max = FLT_MAX;
        if (world.hitall(cur_ray, t_min, t_max, rec)) {
            ray scattered;
            vec3 attenuation;
            switch (rec.mat_type) {
                case 1:
                    if (scatter_lambert(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                        
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0);
                    }
                    break;
                case 2:
                    if (scatter_metal(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0);
                    }
                    break;
                case 3:
                    if (scatter_dielectric(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                        cur_attenuation *= attenuation;
                        cur_ray = scattered;
                        //return cur_attenuation * attenuation * 1.5;
                    }
                    else {
                        return vec3(0.0, 0.0, 0.0);
                    }
                    break;
                default:
                    break;
            }
        }
        else {
            // draw skybox when the ray doesn't hit anything
            vec3 env = sample_env(unit_vector(cur_ray.direction()), envHDR, hdrWidth, hdrHeight);
            return cur_attenuation * env;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}


__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(unsigned int* fb, int max_x, int max_y, int ns, __grid_constant__ const camera cam,
                        __grid_constant__ const hitable_list world, curandState* rand_state, 
                        const float* envHDR, int hdrWidth, int hdrHeight) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = cam.get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state, envHDR, hdrWidth, hdrHeight);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = 255.99 * sqrt(col[0]);
    col[1] = 255.99 * sqrt(col[1]);
    col[2] = 255.99 * sqrt(col[2]);
    fb[pixel_index] = rgbToInt(col[0], col[1], col[2]);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{

#pragma region "gl setup"
    // ------------------------------
    // glfw: initialize and configure
    // ------------------------------

    // not need
    //cudaGLSetGLDevice(0);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Raytrace", NULL, NULL);

    handle = glfwGetWin32Window(window);
    btnWndProc = (WNDPROC)SetWindowLongPtrW(handle, GWLP_WNDPROC, (LONG_PTR)SubclassWindowProc);
    int touch_success = RegisterTouchWindow(handle, 0);

    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // set this to 0, will swap at fullspeed, but app will close very slow, sometime hang
    glfwSwapInterval(1);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // init touch data
    for (int i = 0; i < MAXPOINTS; i++) {
        points[i][0] = -1;
        points[i][1] = -1;
        last_points[i][0] = -1;
        last_points[i][1] = -1;
        diff_points[i][0] = 0;
        diff_points[i][1] = 0;
        idLookup[i] = -1;
    }

    // init skybox data
    int hdrWidth, hdrHeight, hdrChannels;
    float* hdrData = stbi_loadf("overcast_soil_puresky_4k.hdr",
        &hdrWidth,
        &hdrHeight,
        &hdrChannels,
        0);

    if (!hdrData) {
        printf("Failed to load HDR sky!\n");
        return -1;
    }


    // cuda mem for skybox
    float* envHDR_gpu;
    size_t totalBytes = hdrWidth * hdrHeight * hdrChannels * sizeof(float);

    cudaMalloc(&envHDR_gpu, totalBytes);
    cudaMemcpy(envHDR_gpu, hdrData, totalBytes, cudaMemcpyHostToDevice);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    Shader ourShader("tex.vs", "tex.fs");

    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);


    // cuda mem out bind to tex
    // ---------------------------------------
    int num_texels = WIDTH * HEIGHT;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMalloc((void**)&cuda_dest_resource, size_tex_data));

    // create a texture, output from cuda
    glGenTextures(1, &tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

    // fps
    prevTime = glfwGetTime();
#pragma endregion "gl setup"


    // ------------------------------
    // CUDA: RT
    // ------------------------------


    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // create cam
    d_camera.updateCam(0, Yaw, Pitch, lookfrom, lookat, vec3(0, 1, 0), 30.0, float(nx) / float(ny), aperture, dist_to_focus);

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // create a world of metasphere and sphere
    hitable_list a_world;
    world_ptr = &a_world;

    float radius = 4.0f;
    vec3 orbitpivot(0, 1, 0);
    float speed = 1.0f;

    while (!glfwWindowShouldClose(window))//(Msg.message != WM_QUIT)
    {
        currTime = glfwGetTime();
        dt = currTime - prevTime;

        orbitTime += dt;

        world_ptr->meta_world.center[1] =
            vec3(
                radius * cos(orbitTime * speed),
                orbitpivot[1] + 0.0f,
                0.0f
                
                
            );

        // meta-sphere #2 orbits in tilted plane
        world_ptr->meta_world.center[2] =
            vec3(
                radius * cos(orbitTime * speed),
                orbitpivot[1] + radius * sin(orbitTime * speed),
                0.0f
            );

        // update cam
        processInput(window);

        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // begin measure gpu
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        render << <blocks, threads >> > (cuda_dest_resource, nx, ny, ns, d_camera, a_world, d_rand_state, envHDR_gpu, hdrWidth, hdrHeight);

        checkCudaErrors(cudaGetLastError());

        cudaArray* texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));
        checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));

        // end measure gpu
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        ss << elapsedTime << "ms\n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);


        // render gl
        glUniform1i(glGetUniformLocation(ourShader.ID, "texture1"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        // fps
        prevTime = currTime;
    }


    // Free the device memory
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    cudaFree(cuda_dest_resource);
    cudaFree(d_rand_state);
    cudaDeviceReset();

    glfwTerminate();
    return 0;
}
