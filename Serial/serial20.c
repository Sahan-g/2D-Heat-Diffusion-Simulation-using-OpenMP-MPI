#include <stdio.h>
#include <string.h>


#define GRID_HEIGHT 20
#define GRID_WIDTH  20
#define NUM_STEPS   500  

#define D    0.1  
#define DT   1.0   
#define DX   1.0   
#define DY   1.0   

void initialize_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
   
    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            u[i][j] = 0.0;
        }
    }

    int mid   = GRID_HEIGHT / 2;      // =10
    int start = GRID_WIDTH  / 4;      // =5
    int end   = 3 * GRID_WIDTH / 4;   // =15
    for (int j = start; j < end; j++) {
        u[mid][j] = 100.0;
    }
}

void print_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    for (int i = 0; i < GRID_HEIGHT; i++) {
        for (int j = 0; j < GRID_WIDTH; j++) {
            printf("%6.1f ", u[i][j]);
        }
        printf("\n");
    }
}

int main() {
    double u[GRID_HEIGHT][GRID_WIDTH];
    double next_u[GRID_HEIGHT][GRID_WIDTH];

    // initialize
    initialize_grid(u);
    memcpy(next_u, u, sizeof(u));

    printf("Running %d steps on a %dx%d grid (serial)...\n",
           NUM_STEPS, GRID_HEIGHT, GRID_WIDTH);


    print_grid(u);

    // time-stepping loop (serial)
    for (int step = 1; step <= NUM_STEPS; step++) {
        for (int i = 1; i < GRID_HEIGHT - 1; i++) {
            for (int j = 1; j < GRID_WIDTH - 1; j++) {
                double c   = u[i][j];
                double lap = (u[i][j+1] - 2*c + u[i][j-1]) / (DX*DX)
                           + (u[i+1][j] - 2*c + u[i-1][j]) / (DY*DY);
                next_u[i][j] = c + DT * D * lap;
            }
        }
        // copy next_u → u
        memcpy(u, next_u, sizeof(u));
    }

    // output final field
    printf("\nFinal temperature field:\n");
    print_grid(u);

    return 0;
}
