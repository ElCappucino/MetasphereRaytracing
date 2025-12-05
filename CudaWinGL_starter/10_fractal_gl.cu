
//  interop with opengl, glfw
//  use win32 to handle touch input
//  use stb_image for image loading
//  load img, convert to grayscale using cuda, display using opengl




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
#include <time.h>
#include <float.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;
double prevTime = 0.0;
double currTime = 0.0;

// Windows
HWND handle;
WNDPROC currentWndProc;
MSG Msg;
WNDPROC btnWndProc;
std::stringstream ss;

// Touch
#define MAXPOINTS 20
// You will use this array to track touch points
int points[MAXPOINTS][2];
// You will use this array to switch the color / track ids
int idLookup[MAXPOINTS];
int last_points[MAXPOINTS][2];
int diff_points[MAXPOINTS][2];


// cuda opengl interop
GLuint shDrawTex;           // shader
GLuint tex_cudaResult;      // result texture to glBindTexture(GL_TEXTURE_2D, texture);
unsigned int* cuda_dest_resource;  // output from cuda
struct cudaGraphicsResource* cuda_tex_result_resource;

// fractal
#define WIDTH 3800
#define HEIGHT 2100
//double cx = -0.162;
//double cy = 1.04;
////
//double cx = 0.3;
//double cy = -0.01;
//
//double cx = -1.476;
//double cy = 0.0;
////
double cx = -0.79;
double cy = 0.15;
////
//double cx = -0.12;
//double cy = -0.77;
//
//double cx = 0.28;
//double cy = 0.008;
double scale = 1.5;
double panX = 0.0f;
double panY = 0.0f;

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

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        OutputDebugStringA("Key W press \n");



    // 3 point, change to Mandlebrot
    if (points[2][0] >= 0) {
        cx = cx + 0.0001 * scale * diff_points[0][0];
        cy = cy + 0.0001 * scale * diff_points[0][1];
    }
    // 2 point, zoom
    else if (points[1][0] >= 0) {
        if (diff_points[0][1] > 0.01f){
            scale *= 1.1;
        }
        else if (diff_points[0][1] < -0.01f) {
            scale *= 0.9;
        }
    }
    // 1 point, pan
    else if (points[0][0] >= 0) {
        panX = panX + 0.0011 * scale * diff_points[0][0];
        panY = panY + 0.0011 * scale * diff_points[0][1];
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





// ---------------------------------------
// CUDA code
// ---------------------------------------

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        ss << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        OutputDebugStringA(ss.str().c_str());
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

__global__
void img_process(unsigned char* in, unsigned int* out, int width, int height) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 0 || row >= height || col < 0 || col >= width) return;

    int grey_offset = row * width + col;
    int rgb_offset = grey_offset * 3;

    unsigned char r = in[rgb_offset + 0];
    unsigned char g = in[rgb_offset + 1];
    unsigned char b = in[rgb_offset + 2];

    float gray = (0.21f * r + 0.71f * g + 0.07f * b);

    out[row * width + col] = rgbToInt(r,g,b);

}

struct cuComplex {
    double r;
    double i;
    __device__ cuComplex(double a, double b) : r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y, int width, int height, double cx, double cy, double scale, double panX, double panY) {
    //const float scale = 1.5;
    double jx = panX + scale * ((float)(width / 2 - x) / (width / 2));
    double jy = panY + scale * ((float)(height / 2 - y) / (width / 2));
    //cuComplex c(-0.8, 0.156);
    cuComplex c(cx, cy);
    //cuComplex c(jx, jy);		// for Mandelbrot
    cuComplex a(jx, jy);
    int i = 0;
    for (i = 0; i < 1200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i;
    }
    return 0;
}


// 1 block 64 thread, 
__global__ void kernel(unsigned int* out, int width, int height, double cx, double cy, double scale, double panX, double panY) {
    // map from threadIdx/BlockIdx to pixel position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < 0 || row >= height || col < 0 || col >= width) return;

    int offset = row * width + col;
    // now calculate the value at that position
    float juliaValue = julia(col, row, width, height, cx, cy, scale, panX, panY);
    juliaValue = juliaValue / 511.0f;

    unsigned char r = 255 * sqrtf(juliaValue);
    unsigned char g = 255 * powf(juliaValue,3);
    unsigned char b = 0;
    float bf = sinf(2 * 3.14159 * juliaValue);
    if (bf < 0) {
        b = 0;
    }
    else {
        b = 255 * bf;
    }

    out[row * width + col] = rgbToInt(r, g, b);

}


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    // not need
    //cudaGLSetGLDevice(0);

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);

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

    // gl init
    // ---------------------------------------
    Shader ourShader("tex.vs", "tex.fs");
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
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

    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, tex_cudaResult, GL_TEXTURE_2D,cudaGraphicsMapFlagsWriteDiscard));

    // fps
    prevTime = glfwGetTime();
    

    while (!glfwWindowShouldClose(window))//(Msg.message != WM_QUIT)
    {
        // fps
        /*currTime = glfwGetTime();
        double result = currTime - prevTime;
        ss << 1.0f/result << "\n";
        OutputDebugStringA(ss.str().c_str());
        ss.str("");*/

        processInput(window);


        glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // begin measure gpu
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);


        // process img
        int num_thread = 16;
        dim3 dimBlock(num_thread, num_thread, 1);
        dim3 dimGrid(ceil((float)WIDTH / num_thread), ceil((float)HEIGHT / num_thread), 1);
        kernel<<<dimGrid, dimBlock >>> (cuda_dest_resource, WIDTH, HEIGHT, cx, cy, scale, panX, panY);

        //cudaDeviceSynchronize();


        // copy cuda_dest_resource data to the texture
        cudaArray* texture_ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray( &texture_ptr, cuda_tex_result_resource, 0, 0));


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
    cudaFree(cuda_dest_resource);


    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

