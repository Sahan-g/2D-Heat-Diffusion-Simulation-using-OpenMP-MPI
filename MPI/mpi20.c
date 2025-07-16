#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define GRID_HEIGHT 20
#define GRID_WIDTH  20
#define NUM_STEPS   500

#define D    0.1   // Diffusion coefficient
#define DT   1.0   // Time step size
#define DX   1.0   // Spatial step size in x
#define DY   1.0   // Spatial step size in y

// Initialize: zero everywhere, hot line at middle row cols 5…14
void initialize_grid(double u[GRID_HEIGHT][GRID_WIDTH]) {
    for (int i = 0; i < GRID_HEIGHT; i++)
        for (int j = 0; j < GRID_WIDTH; j++)
            u[i][j] = 0.0;
    int mid   = GRID_HEIGHT/2;       // 10
    int start = GRID_WIDTH/4;        // 5
    int end   = 3*GRID_WIDTH/4;      // 15
    for (int j = start; j < end; j++)
        u[mid][j] = 100.0;
}

// For rank 0 to print the gathered full grid
void print_full(int height, int width, double *flat) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%6.6f ", flat[i*width + j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1) Domain decomposition
    int base   = GRID_HEIGHT / size;
    int rem    = GRID_HEIGHT % size;
    int local  = base + (rank < rem ? 1 : 0);
    int offset = rank*base + (rank < rem ? rank : rem);

    // 2) Allocate with 2 ghost rows
    double (*u)[GRID_WIDTH]      = malloc((local+2)*GRID_WIDTH * sizeof(double));
    double (*next_u)[GRID_WIDTH] = malloc((local+2)*GRID_WIDTH * sizeof(double));
    memset(u,      0, (local+2)*GRID_WIDTH*sizeof(double));
    memset(next_u, 0, (local+2)*GRID_WIDTH*sizeof(double));

    // 3) If this process owns the global middle row, set it hot
    int gmid = GRID_HEIGHT/2;
    if (gmid >= offset && gmid < offset+local) {
        int lmid = gmid - offset + 1; // +1 for ghost offset
        for (int j = GRID_WIDTH/4; j < 3*GRID_WIDTH/4; j++)
            u[lmid][j] = 100.0;
    }
    memcpy(next_u, u, (local+2)*GRID_WIDTH*sizeof(double));

    // 4) Prepare for gather: counts and displs
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    int my_count = local * GRID_WIDTH;
    MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; i++)
            displs[i] = displs[i-1] + counts[i-1];
    }

    // 5) Gather and print initial grid on rank 0
    double *flat_initial = NULL;
    if (rank == 0)
        flat_initial = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(double));

    double *my_flat = malloc(local * GRID_WIDTH * sizeof(double));
    for (int i = 0; i < local; i++)
        memcpy(my_flat + i*GRID_WIDTH, u[i+1], GRID_WIDTH * sizeof(double));

    MPI_Gatherv(my_flat, my_count, MPI_DOUBLE,
                flat_initial, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Initial temperature field:\n");
        print_full(GRID_HEIGHT, GRID_WIDTH, flat_initial);
        free(flat_initial);
    }

    // 6) Create row datatype
    MPI_Datatype row_t;
    MPI_Type_contiguous(GRID_WIDTH, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // 7) Time stepping with ghost exchanges
    double t0 = MPI_Wtime();
    for (int step = 1; step <= NUM_STEPS; step++) {
        int up   = (rank == 0        ? MPI_PROC_NULL : rank-1);
        int down = (rank == size-1   ? MPI_PROC_NULL : rank+1);

        // exchange top ghost
        MPI_Sendrecv(u[1],       1, row_t, up,   0,
                     u[0],       1, row_t, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // exchange bottom ghost
        MPI_Sendrecv(u[local],   1, row_t, down, 1,
                     u[local+1], 1, row_t, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // determine local update bounds to preserve global Dirichlet boundaries
        int i_start = 1;
        int i_end   = local;
        // if this chunk starts at global row 0, skip local i=1
        if (offset == 0)          i_start = 2;
        // if this chunk ends at global last row, skip local i=local
        if (offset + local == GRID_HEIGHT) i_end = local - 1;

        // compute interior
        for (int i = i_start; i <= i_end; i++) {
            for (int j = 1; j < GRID_WIDTH - 1; j++) {
                double c   = u[i][j];
                double lap = (u[i][j+1] - 2*c + u[i][j-1]) / (DX*DX)
                           + (u[i+1][j] - 2*c + u[i-1][j]) / (DY*DY);
                next_u[i][j] = c + DT * D * lap;
            }
        }
        // copy all including ghosts
        memcpy(u, next_u, (local+2)*GRID_WIDTH*sizeof(double));
    }
    double t1 = MPI_Wtime();

    // 8) Gather and print final grid on rank 0
    double *flat_final = NULL;
    if (rank == 0)
        flat_final = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(double));

    for (int i = 0; i < local; i++)
        memcpy(my_flat + i*GRID_WIDTH, u[i+1], GRID_WIDTH * sizeof(double));

    MPI_Gatherv(my_flat, my_count, MPI_DOUBLE,
                flat_final, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nFinal temperature field:\n");
        print_full(GRID_HEIGHT, GRID_WIDTH, flat_final);
        printf("\nMPI (%d ranks) elapsed: %.6f s\n", size, t1 - t0);
        free(flat_final);
        free(counts);
        free(displs);
    }

    // 9) Cleanup
    free(my_flat);
    MPI_Type_free(&row_t);
    free(u);
    free(next_u);
    MPI_Finalize();
    return 0;
}
