#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// — Test parameters —
#define GRID_HEIGHT 20
#define GRID_WIDTH  20
#define NUM_STEPS   500

#define D    0.1   // diffusion coefficient
#define DT   1.0   // time step
#define DX   1.0   // Δx
#define DY   1.0   // Δy

// Print a flat GRID_HEIGHT×GRID_WIDTH array
void print_full(int H, int W, double *flat) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            printf("%6.6f ", flat[i * W + j]);
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
    // Global index of my first real row:
    int start  = rank * base + (rank < rem ? rank : rem);

    // 2) Allocate with 2 ghost rows
    double (*u)[GRID_WIDTH]      = malloc((local + 2) * GRID_WIDTH * sizeof(double));
    double (*next_u)[GRID_WIDTH] = malloc((local + 2) * GRID_WIDTH * sizeof(double));

    // 3) Initialize to zero & hot mid‑line
    for (int i = 0; i < local + 2; i++)
        for (int j = 0; j < GRID_WIDTH; j++)
            u[i][j] = 0.0;
    int gmid = GRID_HEIGHT / 2;  // 10
    if (gmid >= start && gmid < start + local) {
        int lmid = gmid - start + 1;      // +1 for top ghost
        for (int j = GRID_WIDTH / 4; j < 3 * GRID_WIDTH / 4; j++)
            u[lmid][j] = 100.0;
    }
    memcpy(next_u, u, (local + 2) * GRID_WIDTH * sizeof(double));

    // 4) Prepare gather counts/displs
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    int my_count = local * GRID_WIDTH;
    MPI_Gather(&my_count, 1, MPI_INT,
               counts,  1, MPI_INT,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < size; r++)
            displs[r] = displs[r - 1] + counts[r - 1];
    }

    // 5) Gather & print initial grid on rank 0
    double *flat_init = NULL;
    if (rank == 0)
        flat_init = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(double));
    double *my_flat = malloc(local * GRID_WIDTH * sizeof(double));
    for (int i = 0; i < local; i++)
        memcpy(my_flat + i * GRID_WIDTH,
               u[i + 1],
               GRID_WIDTH * sizeof(double));
    MPI_Gatherv(my_flat, my_count, MPI_DOUBLE,
                flat_init, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Initial temperature field:\n");
        print_full(GRID_HEIGHT, GRID_WIDTH, flat_init);
        free(flat_init);
    }

    // 6) MPI row type for ghost exchange
    MPI_Datatype row_t;
    MPI_Type_contiguous(GRID_WIDTH, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    double t0 = MPI_Wtime();

    // 7) Time‑stepping
    for (int step = 1; step <= NUM_STEPS; step++) {
        int up   = (rank == 0       ? MPI_PROC_NULL : rank - 1);
        int down = (rank == size-1  ? MPI_PROC_NULL : rank + 1);

        // exchange top ghost
        MPI_Sendrecv(u[1],      1, row_t, up,   0,
                     u[0],      1, row_t, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // exchange bottom ghost
        MPI_Sendrecv(u[local],  1, row_t, down, 1,
                     u[local+1],1, row_t, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // determine which rows to compute (skip global boundary)
        int i0 = 1, i1 = local;
        if (start == 0)               i0 = 2;       // skip global row 0
        if (start + local == GRID_HEIGHT) i1 = local - 1;  // skip last row

        // OpenMP‑parallel stencil
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int i = i0; i <= i1; i++) {
                for (int j = 1; j < GRID_WIDTH - 1; j++) {
                    double c    = u[i][j];
                    double lapx = (u[i][j+1] - 2*c + u[i][j-1]) / (DX*DX);
                    double lapy = (u[i+1][j] - 2*c + u[i-1][j]) / (DY*DY);
                    next_u[i][j] = c + DT * D * (lapx + lapy);
                }
            }
            #pragma omp single
            memcpy(u, next_u, (local + 2) * GRID_WIDTH * sizeof(double));
        }
    }

    double t1 = MPI_Wtime();

    // 8) Collect final grid: rebuild my_flat from updated u
    for (int i = 0; i < local; i++)
        memcpy(my_flat + i * GRID_WIDTH,
               u[i + 1],
               GRID_WIDTH * sizeof(double));

    double *flat_fin = NULL;
    if (rank == 0)
        flat_fin = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(double));
    MPI_Gatherv(my_flat, my_count, MPI_DOUBLE,
                flat_fin, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nFinal temperature field:\n");
        print_full(GRID_HEIGHT, GRID_WIDTH, flat_fin);
        printf("\nHybrid MPI+OpenMP (%d ranks × %d threads) in %.3f s\n",
               size, omp_get_max_threads(), t1 - t0);
        free(flat_fin);
        free(counts);
        free(displs);
    }

    // 9) Cleanup
    MPI_Type_free(&row_t);
    free(u);
    free(next_u);
    free(my_flat);
    MPI_Finalize();
    return 0;
}
