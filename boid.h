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
void update_boid_position(Boid* b, Boid* boids, int N,float perception_radius,float fov_deg,float w_align,float w_cohesion,float w_separation);