#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>   

// ---Parameters ---
#define GRID_HEIGHT 1500
#define GRID_WIDTH  1500
#define NUM_STEPS   500

#define D    0.1 // Diffusion coefficient
#define DT   1.0 // Time step size
#define DX   1.0 // Spatial step size in x
#define DY   1.0 // Spatial step size in y

void initialize_grid(double u[GRID_HEIGHT][GRID_WIDTH]);
void print_grid(double u[GRID_HEIGHT][GRID_WIDTH]);

int main() {
    struct timespec start_time, end_time;
    double elapsed_time;

    double (*u)[GRID_WIDTH] = malloc(sizeof(double[GRID_HEIGHT][GRID_WIDTH]));
    double (*next_u)[GRID_WIDTH] = malloc(sizeof(double[GRID_HEIGHT][GRID_WIDTH]));

    initialize_grid(u);
    memcpy(next_u, u, sizeof(double[GRID_HEIGHT][GRID_WIDTH]));

    printf("Starting simulation...\n");
    printf("Grid Size: %d x %d\n", GRID_HEIGHT, GRID_WIDTH);
    printf("Time Steps: %d\n", NUM_STEPS);
    printf("----------------------------------\n");
    //print_grid(u);

    // --- Start Timer ---
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // ---Simulation---
    for (int step = 1; step <= NUM_STEPS; step++) {
        for (int i = 1; i < GRID_HEIGHT - 1; i++) {
            for (int j = 1; j < GRID_WIDTH - 1; j++) {
                double center = u[i][j];
                double up     = u[i - 1][j];
                double down   = u[i + 1][j];
                double left   = u[i][j - 1];
                double right  = u[i][j + 1];

                double d2u_dx2 = (right - 2 * center + left) / (DX * DX);
                double d2u_dy2 = (down  - 2 * center + up)   / (DY * DY);

                next_u[i][j] = center + DT * D * (d2u_dx2 + d2u_dy2);
            }
        }
        memcpy(u, next_u, sizeof(double[GRID_HEIGHT][GRID_WIDTH]));
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    elapsed_time = (end_time.tv_sec - start_time.tv_sec);
    elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1.0e9; // 1.0e9 is 1 billion

    printf("\n--- Simulation Complete ---\n");
    printf("Total execution time: %.6f seconds\n", elapsed_time);
    //print_grid(u);

    free(u);
    free(next_u);

    return 0;
}


void initialize_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            u[i][j] = 0.0;
        }
    }
    int middle_row = GRID_HEIGHT / 2;
    for (int j = GRID_WIDTH / 4; j < 3 * GRID_WIDTH / 4; j++) {
        u[middle_row][j] = 100.0;
    }
}

void print_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            printf("%8.3f ", u[i][j]);
        }
        printf("\n");
    }
}