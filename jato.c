#include <stdio.h>
#include <SDL2/SDL.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "boid.h"
#include <SDL2/SDL_ttf.h>

#define MIN_SPEED 1.0f
#define MAX_SPEED 4.0f
#define FIXED_DT (1.0f / 60.0f)


void draw_triangle(SDL_Renderer* renderer, Point p1, Point p2, Point p3);
void draw_boid(SDL_Renderer* renderer, Boid* b, float size);
void initialize_boids(Boid* boids, int N, int width, int height);
void update_boid(Boid* b, Boid* boids,Boid* neighbors, int N, int width, int height,float perception_radius, float fov_deg,
                 float w_align, float w_cohesion, float w_separation);
void draw_fov(SDL_Renderer* renderer, Boid* b, float perception_radius, float fov_deg);
void draw_text(SDL_Renderer* renderer, TTF_Font* font, const char* text, int x, int y);
void handle_keyDown(SDL_Keycode key);
bool running = true;
float w_align = 0.0f;
float w_cohesion = 0.0f;
float w_separation = 0.0f;
float fov_deg = 270.0f;
float perception_radius = 50.0f;
float delta = 0.1f, delta_angle = 5.0f, delta_radius = 5.0f;
bool show_fov = false;



int main(int argc, char* argv[]) {

    if (argc < 7) {
        printf("Usage: %s <num_boids> <perception_radius> <fov_deg> <w_align> <w_cohesion> <w_separation>\n", argv[0]);
        return 1;
    }

    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();

    int window_width = 1400;
    int window_height = 1000;

    SDL_Window* window = SDL_CreateWindow(
        "Boids",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        window_width, window_height,
        0
    );

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    TTF_Font* font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16);

    int N = atoi(argv[1]);
    perception_radius = atof(argv[2]);
    fov_deg = atof(argv[3]);
    w_align = atof(argv[4]);
    w_cohesion = atof(argv[5]);
    w_separation = atof(argv[6]);

    Boid* boids = malloc(sizeof(Boid) * N);
    Boid* neighbors = malloc(sizeof(Boid) * N);

    initialize_boids(boids, N, window_width, window_height);
    int random_boid_index = rand() % N;

    SDL_Event event;

    /* TIMING */
    Uint64 now = SDL_GetPerformanceCounter();
    Uint64 last = now;
    double accumulator = 0.0;

    /* FPS */
    int frames = 0;
    float fps = 0.0f;
    Uint32 fps_timer = SDL_GetTicks();
    while (running) {

        now = SDL_GetPerformanceCounter();
        double frameTime = (double)(now - last) / SDL_GetPerformanceFrequency();
        last = now;
        accumulator += frameTime;

        /* INPUT */
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN)
                handle_keyDown(event.key.keysym.sym);
        }

        /* FIXED UPDATE */
        while (accumulator >= FIXED_DT) {
            for (int i = 0; i < N; i++) {
                update_boid(&boids[i], boids, neighbors, N,
                            window_width, window_height,
                            perception_radius, fov_deg,
                            w_align, w_cohesion, w_separation);
            }
            accumulator -= FIXED_DT;
        }

        /* RENDER */
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        for (int i = 0; i < N; i++) {
            draw_boid(renderer, &boids[i], 10.0f);
            if (i == random_boid_index && show_fov)
                draw_fov(renderer, &boids[i], perception_radius, fov_deg);
        }

        char info[256];
        snprintf(info, sizeof(info),
            "Align %.2f | Cohesion %.2f | Sep %.2f | FOV %.0f | R %.0f",
            w_align, w_cohesion, w_separation, fov_deg, perception_radius);

        draw_text(renderer, font, info, 10, 10);

        frames++;
        if (SDL_GetTicks() - fps_timer >= 1000) {
            fps = frames;
            frames = 0;
            fps_timer = SDL_GetTicks();
        }

        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.0f", fps);
        draw_text(renderer, font, fps_text, window_width - 140, window_height - 30);

        SDL_RenderPresent(renderer);
    }

    free(boids);
    free(neighbors);
    TTF_Quit();
    SDL_Quit();
    return 0;
}
void draw_text(SDL_Renderer* renderer, TTF_Font* font, const char* text, int x, int y) {
    SDL_Color color = {255, 255, 255, 255};
    SDL_Surface* surface = TTF_RenderText_Solid(font, text, color);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);

    SDL_Rect dest = { x, y, surface->w, surface->h };
    SDL_RenderCopy(renderer, texture, NULL, &dest);

    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
}

void handle_keyDown(SDL_Keycode key) {
     switch(key) {
                case SDLK_ESCAPE: running = false; break;

                // Alignment
                case SDLK_w: w_align += delta; break;
                case SDLK_s: w_align -= delta; break;

                // Cohesion
                case SDLK_a: w_cohesion += delta; break;
                case SDLK_d: w_cohesion -= delta; break;

                // Separation
                case SDLK_z: w_separation += delta; break;
                case SDLK_x: w_separation -= delta; break;

                // FOV
                case SDLK_q: fov_deg += delta_angle; break;
                case SDLK_e: fov_deg -= delta_angle; break;

                // Perception radius
                case SDLK_r: perception_radius += delta_radius; break;
                case SDLK_f: perception_radius -= delta_radius; break;
                case SDLK_c: show_fov = !show_fov; break;
                default:
                    break;
            }
}
void draw_fov(SDL_Renderer* renderer, Boid* b, float perception_radius, float fov_deg) {
    float cx = b->position.x;
    float cy = b->position.y;

    // normalizacija velocity
    float vx = b->velocity.x;
    float vy = b->velocity.y;
    float len = sqrtf(vx*vx + vy*vy);
    if (len == 0) { vx = 0; vy = -1; }
    else { vx /= len; vy /= len; }

    // racunanje putanje
    float heading = atan2f(vy, vx);

    float half_fov_rad = fov_deg * 0.5f * (3.14159265f / 180.0f);

    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);

    int segments = 50;
    //crtanje fov
    for (int i = 0; i < segments; i++) {
        float t1 = (float)i / segments;
        float t2 = (float)(i+1) / segments;

        float angle1 = heading - half_fov_rad + t1 * fov_deg * (3.14159265f / 180.0f);
        float angle2 = heading - half_fov_rad + t2 * fov_deg * (3.14159265f / 180.0f);

        float ax1 = cx + cosf(angle1) * perception_radius;
        float ay1 = cy + sinf(angle1) * perception_radius;

        float ax2 = cx + cosf(angle2) * perception_radius;
        float ay2 = cy + sinf(angle2) * perception_radius;

        SDL_RenderDrawLine(renderer, (int)cx, (int)cy, (int)ax1, (int)ay1);
        SDL_RenderDrawLine(renderer, (int)ax1, (int)ay1, (int)ax2, (int)ay2);
    }
}


void initialize_boids(Boid* boids, int N, int width, int height) {
    srand((unsigned int)time(NULL));

    //nasumične pozicije i brzine
    for (int i = 0; i < N; i++) {
        boids[i].position = create_point(rand() % width, rand() % height);

        float angle = ((float)rand() / RAND_MAX) * 2.0f * 3.14159265f;
        float speed = MIN_SPEED + ((float)rand() / RAND_MAX) * (MAX_SPEED - MIN_SPEED);

        boids[i].velocity = create_vector(cosf(angle) * speed, sinf(angle) * speed);
        boids[i].acceleration = create_vector(0.0f, 0.0f);
    }
}

void update_boid(Boid* b, Boid* boids, Boid* neighbors, int N, int width, int height,
                 float perception_radius, float fov_deg,
                 float w_align, float w_cohesion, float w_separation) {

    // ažuriranje pozicije na osnovu pravila jata
    update_boid_position(b, boids,neighbors, N, perception_radius, fov_deg, w_align, w_cohesion, w_separation);

    // wrapping oko ivica ekrana
    if (b->position.x < 0) b->position.x += width;
    if (b->position.x >= width) b->position.x -= width;
    if (b->position.y < 0) b->position.y += height;
    if (b->position.y >= height) b->position.y -= height;

    // resetovanje ubrzanja za sledeći frame
    b->acceleration.x = 0.0f;
    b->acceleration.y = 0.0f;

}
void draw_triangle(SDL_Renderer* renderer, Point p1, Point p2, Point p3) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawLine(renderer, (int)p1.x, (int)p1.y, (int)p2.x, (int)p2.y);
    SDL_RenderDrawLine(renderer, (int)p2.x, (int)p2.y, (int)p3.x, (int)p3.y);
    SDL_RenderDrawLine(renderer, (int)p3.x, (int)p3.y, (int)p1.x, (int)p1.y);
}
void draw_boid(SDL_Renderer* renderer, Boid* b, float size) {
    // normalizacija velocity
    float vx = b->velocity.x;
    float vy = b->velocity.y;
    float len = sqrtf(vx*vx + vy*vy);
    if (len == 0) {
        vx = 0; vy = -1;
    } else {
        vx /= len;
        vy /= len;
    }

    float px = -vy;
    float py = vx;

    Point front = {
        b->position.x + vx * size,
        b->position.y + vy * size
    };
    Point left = {
        b->position.x - vx * size * 0.5f + px * size * 0.5f,
        b->position.y - vy * size * 0.5f + py * size * 0.5f
    };
    Point right = {
        b->position.x - vx * size * 0.5f - px * size * 0.5f,
        b->position.y - vy * size * 0.5f - py * size * 0.5f
    };
    //crtanje trougla boida
    draw_triangle(renderer, front, left, right);
}