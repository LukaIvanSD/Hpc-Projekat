// boid_cuda.cu
#include <cuda_runtime.h>
#include "boid.h"
#include <stdio.h>
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }else{\
      fprintf(stderr, "CUDA Runtime Success: %s:%i:%d = %s\n", \
              __FILE__,                         \
              __LINE__,                         \
              result,\
              cudaGetErrorString(result));\
    }                                             \
} while(0)

__constant__ int c_width;
__constant__ int c_height;
__device__ __constant__ float PI = 3.14159265f;
__device__ __constant__ float MAX_SPEED = 4.0f;


Boid* d_boids;


__device__ void update_boid_position_device(Boid* b, Boid* boids, int N,
                                            float perception_radius, float fov_deg,
                                            float w_align, float w_cohesion, float w_separation)
{

    Vector align = {0.0f, 0.0f};
    Vector coh   = {0.0f, 0.0f};
    Vector sep   = {0.0f, 0.0f};

    int count = 0;
    float radius2 = perception_radius * perception_radius;
    float half_fov = fov_deg * 0.5f * (PI / 180.0f);

    // normalizacija velocity boida b
    float vlen = sqrtf(b->velocity.x*b->velocity.x + b->velocity.y*b->velocity.y);
    if (vlen < 1e-5f) vlen = 1e-5f;

    float vx = b->velocity.x / vlen;
    float vy = b->velocity.y / vlen;

    for (int i = 0; i < N; i++) {
        if (&boids[i] == b) continue;

        float dx = boids[i].position.x - b->position.x;
        float dy = boids[i].position.y - b->position.y;
        float dist2 = dx*dx + dy*dy;

        if (dist2 > radius2 || dist2 == 0.0f) continue;

        float dist = sqrtf(dist2);
        float ndx = dx / dist;
        float ndy = dy / dist;

        float dot = vx*ndx + vy*ndy;
        dot = fminf(1.0f, fmaxf(-1.0f, dot));   // CLAMP
        float angle = acosf(dot);

        if (angle > half_fov) continue;

        // ALIGNMENT
        align.x += boids[i].velocity.x;
        align.y += boids[i].velocity.y;

        // COHESION
        coh.x += boids[i].position.x;
        coh.y += boids[i].position.y;

        // SEPARATION
        sep.x -= dx / dist2;
        sep.y -= dy / dist2;

        count++;
    }

    if (count > 0) {
        align.x /= count;
        align.y /= count;

        coh.x = (coh.x / count) - b->position.x;
        coh.y = (coh.y / count) - b->position.y;
    }

    // ACCELERATION
    b->acceleration.x =
        align.x * w_align +
        coh.x   * w_cohesion +
        sep.x   * w_separation;

    b->acceleration.y =
        align.y * w_align +
        coh.y   * w_cohesion +
        sep.y   * w_separation;

    // VELOCITY UPDATE
    b->velocity.x += b->acceleration.x;
    b->velocity.y += b->acceleration.y;

    float speed = sqrtf(b->velocity.x*b->velocity.x + b->velocity.y*b->velocity.y);
    if (speed > MAX_SPEED) {
        b->velocity.x = (b->velocity.x / speed) * MAX_SPEED;
        b->velocity.y = (b->velocity.y / speed) * MAX_SPEED;
    }

    // POSITION UPDATE
    b->position.x += b->velocity.x;
    b->position.y += b->velocity.y;


    if (b->position.x < 0) b->position.x += c_width;
    if (b->position.x >= c_width) b->position.x -= c_width;
    if (b->position.y < 0) b->position.y += c_height;
    if (b->position.y >= c_height) b->position.y -= c_height;

}

__global__ void update_boids_kernel(Boid* boids, int N,
                                    float perception_radius, float fov_deg,
                                    float w_align, float w_cohesion, float w_separation)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    Boid b = boids[i];
    if (i >= N) return;

    update_boid_position_device(&b, boids, N,
                                perception_radius, fov_deg,
                                w_align, w_cohesion, w_separation);
    boids[i] = b;
}




extern "C"
void cuda_set_simulation_constants(int width, int height)
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_height, &height, sizeof(int)));
}
extern "C"
void cuda_init_boids(Boid** boids, int N) {
    CUDA_CHECK(cudaMallocHost(boids, N * sizeof(Boid)));
    CUDA_CHECK(cudaMalloc(&d_boids, N * sizeof(Boid)));
}
extern "C"
void cuda_copy_boids_to_device(Boid* boids, int N) {
    CUDA_CHECK(cudaMemcpy(d_boids, boids, N * sizeof(Boid), cudaMemcpyHostToDevice));
}

extern "C"
void cuda_update_boids(Boid* boids, int N,
                       float perception_radius, float fov_deg,
                       float w_align, float w_cohesion, float w_separation)
{
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    update_boids_kernel<<<blocks, threads>>>(d_boids, N,
                                             perception_radius, fov_deg,
                                             w_align, w_cohesion, w_separation);

    cudaGetLastError();

    cudaDeviceSynchronize();

    cudaMemcpy(boids, d_boids, N * sizeof(Boid), cudaMemcpyDeviceToHost);
}

extern "C"
void cuda_shutdown() {
    if (d_boids) cudaFree(d_boids);
  }
