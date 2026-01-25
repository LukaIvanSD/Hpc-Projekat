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
Vector calculate_alignment(Boid* b, Boid* boids, int N) {
    Vector avg = {0.0f, 0.0f};

    if (N == 0) return avg;

    for (int i = 0; i < N; i++) {
        avg.x += boids[i].velocity.x;
        avg.y += boids[i].velocity.y;
    }

    avg.x /= N;
    avg.y /= N;

    return avg;
}
Vector calculate_cohesion(Boid* b, Boid* boids, int N) {
    Vector center = {0.0f, 0.0f};
    Vector steering = {0.0f, 0.0f};

    if (N == 0) return steering;

    //Nalazenje centra mase suseda
    for (int i = 0; i < N; i++) {
        center.x += boids[i].position.x;
        center.y += boids[i].position.y;
    }
    center.x /= N;
    center.y /= N;

    //Vektor ka centru mase
    steering.x = center.x - b->position.x;
    steering.y = center.y - b->position.y;

    return steering;
}
Vector calculate_separation(Boid* b, Boid* boids, int N) {
    Vector steering = {0.0f, 0.0f};

    if (N == 0) return steering;

    for (int i = 0; i < N; i++) {
        float dx = b->position.x - boids[i].position.x;
        float dy = b->position.y - boids[i].position.y;
        float dist2 = dx*dx + dy*dy;

        if (dist2 > 0) {
            steering.x += dx / dist2;
            steering.y += dy / dist2;
        }
    }

    return steering;
}

void find_neighbors(Boid* b, Boid* boids, Boid* neighbors, int N, float perception_radius, float fov_deg, int* neighbor_count) {
    int count = 0;
    float radius2 = perception_radius * perception_radius;
    float half_fov = fov_deg * 0.5f * (M_PI / 180.0f);

    for (int i = 0; i < N; i++) {
        if (&boids[i] == b) continue; // preskoči sebe

        // vektor od b ka susedu
        float dx = boids[i].position.x - b->position.x;
        float dy = boids[i].position.y - b->position.y;
        float dist2 = dx*dx + dy*dy;

        if (dist2 > radius2) continue; // van vidokruga

        // normalizacija vektora do suseda
        float len = sqrtf(dist2);
        float ndx = dx / len;
        float ndy = dy / len;

        // normalizacija velocity boida
        float vlen = sqrtf(b->velocity.x*b->velocity.x + b->velocity.y*b->velocity.y);
        if (vlen == 0.0f) vlen = 0.0001f;

        float vx = b->velocity.x / vlen;
        float vy = b->velocity.y / vlen;

        // račun kosinusa ugla između velocity i vektora do suseda
        float dot = vx*ndx + vy*ndy;
        float angle = acosf(dot);

        if (angle <= half_fov) {
            neighbors[count++] = boids[i]; //sused pronađen
        }
    }

    *neighbor_count = count;
}


void update_boid_position(Boid* b, Boid* all_boids,Boid* neighbors, int N,
                          float perception_radius, float fov_deg,
                          float w_align, float w_cohesion, float w_separation) {
    int n_neighbors = 0;

    //Pronađi susede
    find_neighbors(b, all_boids, neighbors, N, perception_radius, fov_deg, &n_neighbors);

    //Izračunaj pravila
    Vector align = calculate_alignment(b, neighbors, n_neighbors);
    Vector coh   = calculate_cohesion(b, neighbors, n_neighbors);
    Vector sep   = calculate_separation(b, neighbors, n_neighbors);

    //Kombinacija pravila u acceleration
    b->acceleration.x = align.x * w_align + coh.x * w_cohesion + sep.x * w_separation;
    b->acceleration.y = align.y * w_align + coh.y * w_cohesion + sep.y * w_separation;

    //Update velocity
    b->velocity.x += b->acceleration.x;
    b->velocity.y += b->acceleration.y;

    //Limit maksimalnu brzinu
    float speed = sqrtf(b->velocity.x*b->velocity.x + b->velocity.y*b->velocity.y);
    if (speed > MAX_SPEED) {
        b->velocity.x = (b->velocity.x / speed) * MAX_SPEED;
        b->velocity.y = (b->velocity.y / speed) * MAX_SPEED;
    }

    //Update position
    b->position.x += b->velocity.x;
    b->position.y += b->velocity.y;
}

