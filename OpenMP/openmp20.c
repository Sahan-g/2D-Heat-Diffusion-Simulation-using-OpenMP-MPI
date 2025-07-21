#include <stdio.h>
#include <string.h>
#include <omp.h>

// — Test parameters —
#define GRID_HEIGHT 20
#define GRID_WIDTH  20
#define NUM_STEPS   500  // you can reduce or increase this

#define D    0.1   // diffusion coefficient
#define DT   1.0   // time step
#define DX   1.0   // grid spacing x
#define DY   1.0   // grid spacing y

void initialize_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    // zero everything
    for (int i = 0; i < GRID_HEIGHT; i++)
        for (int j = 0; j < GRID_WIDTH; j++)
            u[i][j] = 0.0;

    // small “hot” line in the middle row
    int mid = GRID_HEIGHT / 2;            // =10
    int start = GRID_WIDTH / 4;           // =5
    int end   = 3 * GRID_WIDTH / 4;       // =15
    for (int j = start; j < end; j++)
        u[mid][j] = 100.0;
}

void print_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            printf("%6.6f ", u[i][j]);
        }
        printf("\n");
    }
}

int main() {
    double u[GRID_HEIGHT][GRID_WIDTH];
    double next_u[GRID_HEIGHT][GRID_WIDTH];

    initialize_grid(u);
    memcpy(next_u, u, sizeof(u));

    printf("Running %d steps on a %dx%d grid...\n",
           NUM_STEPS, GRID_HEIGHT, GRID_WIDTH);

    print_grid(u);

    #pragma omp parallel
    for (int step = 1; step <= NUM_STEPS; step++) {
        #pragma omp for collapse(2)
        for (int i = 1; i < GRID_HEIGHT - 1; i++) {
            for (int j = 1; j < GRID_WIDTH - 1; j++) {
                double c = u[i][j];
                double lap = (u[i][j+1] - 2*c + u[i][j-1])/(DX*DX)
                           + (u[i+1][j] - 2*c + u[i-1][j])/(DY*DY);
                next_u[i][j] = c + DT * D * lap;
            }
        }
        #pragma omp single
        memcpy(u, next_u, sizeof(u));
    }

    printf("\nFinal temperature field:\n");
    print_grid(u);
    return 0;
}
