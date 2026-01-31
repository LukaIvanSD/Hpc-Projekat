#include "boid.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define MAX_SPEED 4.0f
Point create_point(float x, float y) {
    return (Point){ x, y };
}
Vector create_vector(float x, float y) {
    return (Vector){ x, y };
}

void update_boid_position(Boid* b, Boid* all_boids, int N,
                          float perception_radius, float fov_deg,
                          float w_align, float w_cohesion, float w_separation) {
    Vector align = {0.0f, 0.0f};
    Vector coh   = {0.0f, 0.0f};
    Vector sep   = {0.0f, 0.0f};

    int count = 0;
    float radius2 = perception_radius * perception_radius;
    float half_fov = fov_deg * 0.5f * (M_PI / 180.0f);

    // normalizacija velocity boida b
    float vlen = sqrtf(b->velocity.x*b->velocity.x + b->velocity.y*b->velocity.y);
    if (vlen < 1e-5f) vlen = 1e-5f;

    float vx = b->velocity.x / vlen;
    float vy = b->velocity.y / vlen;

    for (int i = 0; i < N; i++) {
        if (&all_boids[i] == b) continue;

        float dx = all_boids[i].position.x - b->position.x;
        float dy = all_boids[i].position.y - b->position.y;
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
        align.x += all_boids[i].velocity.x;
        align.y += all_boids[i].velocity.y;

        // COHESION
        coh.x += all_boids[i].position.x;
        coh.y += all_boids[i].position.y;

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
}