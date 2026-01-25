typedef struct {
    float x;
    float y;
} Point;

typedef struct {
    float x;
    float y;
} Vector;

typedef struct {
    Point position;
    Vector velocity;
    Vector acceleration;
} Boid;

Boid create_boid(Point position, Vector velocity);
Point create_point(float x, float y);
Vector create_vector(float x, float y);
Vector calculate_alignment(Boid* b, Boid* boids, int N);
Vector calculate_cohesion(Boid* b, Boid* boids, int N);
Vector calculate_separation(Boid* b, Boid* boids, int N);
void find_neighbors(Boid* b, Boid* boids,Boid* neighbors, int N, float perception_radius,float fov, int* neighbor_count);
void update_boid_position(Boid* b, Boid* boids,Boid* neighbors, int N,float perception_radius,float fov_deg,float w_align,float w_cohesion,float w_separation);