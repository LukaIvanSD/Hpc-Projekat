
void cuda_init_boids(Boid** boids, int N);
void cuda_update_boids( Boid* boids, int N, float perception_radius, float fov_deg,
                       float w_align, float w_cohesion, float w_separation);
void cuda_shutdown();
void cuda_copy_boids_to_device(Boid* boids, int N);
void cuda_set_simulation_constants(int width, int height, float perception_radius);
void cuda_update_optimized_boids( Boid* boids, int N, float perception_radius, float fov_deg,
                               float w_align, float w_cohesion, float w_separation);
