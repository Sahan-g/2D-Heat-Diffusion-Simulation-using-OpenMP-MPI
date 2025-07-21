#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// ---Parameters ---
#define GRID_HEIGHT 1500
#define GRID_WIDTH  1500
#define NUM_STEPS   2000

#define D    0.1  // Diffusion coefficient
#define DT   1.0  // Time step size
#define DX   1.0  // Spatial step size in x
#define DY   1.0  // Spatial step size in y

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int base_rows  = GRID_HEIGHT / size;
    int remainder  = GRID_HEIGHT % size;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int start_row  = rank * base_rows + (rank < remainder ? rank : remainder);

    //  allocate local chunk + 2 ghost rows
    double (*u)[GRID_WIDTH]      = malloc((local_rows+2)*GRID_WIDTH*sizeof(double));
    double (*next_u)[GRID_WIDTH] = malloc((local_rows+2)*GRID_WIDTH*sizeof(double));
    if (!u || !next_u) MPI_Abort(MPI_COMM_WORLD, 1);

    //  initialize
    for (int i = 0; i < local_rows+2; i++)
        for (int j = 0; j < GRID_WIDTH; j++)
            u[i][j] = 0.0;
    int global_mid = GRID_HEIGHT/2;
    if (global_mid >= start_row && global_mid < start_row+local_rows) {
        int local_mid = global_mid - start_row + 1;
        int js = GRID_WIDTH/4, je = 3*GRID_WIDTH/4;
        for (int j = js; j < je; j++)
            u[local_mid][j] = 100.0;
    }
    memcpy(next_u, u, (local_rows+2)*GRID_WIDTH*sizeof(double));

    // define rowtype to send ghost row
    MPI_Datatype row_t;
    MPI_Type_contiguous(GRID_WIDTH, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    double t0 = MPI_Wtime();

    // time‐stepping
    for (int step = 1; step <= NUM_STEPS; step++) {
        int up   = (rank == 0        ? MPI_PROC_NULL : rank-1);
        int down = (rank == size-1   ? MPI_PROC_NULL : rank+1);

        MPI_Sendrecv(u[1],      1, row_t, up,   0,
                     u[0],      1, row_t, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u[local_rows], 1, row_t, down, 1,
                     u[local_rows+1],1, row_t, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int i0 = 1, i1 = local_rows;
        if (start_row == 0)                i0 = 2;
        if (start_row + local_rows == GRID_HEIGHT) i1 = local_rows - 1;

        for (int i = i0; i <= i1; i++) {
            for (int j = 1; j < GRID_WIDTH-1; j++) {
                double c    = u[i][j];
                double lapx = (u[i][j+1] - 2*c + u[i][j-1])/(DX*DX);
                double lapy = (u[i+1][j] - 2*c + u[i-1][j])/(DY*DY);
                next_u[i][j] = c + DT * D * (lapx + lapy);
            }
        }
        memcpy(u, next_u, (local_rows+2)*GRID_WIDTH*sizeof(double));
    }

    double t1 = MPI_Wtime();
    if (rank == 0) {
        printf("MPI simulation (%d ranks) complete in %.6f s\n",
               size, t1 - t0);
    }
    // memory allocation for fianal grid
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
    }
    int count_ = local_rows * GRID_WIDTH;
    MPI_Gather(&count_, 1, MPI_INT,
               counts,  1, MPI_INT,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < size; r++)
            displs[r] = displs[r-1] + counts[r-1];
    }

    //  Flatten local data (sikp ghost rows)
    double *flat = malloc(count_ * sizeof(double));
    for (int i = 0; i < local_rows; i++) {
        memcpy(flat + i*GRID_WIDTH,
               u[i+1],
               GRID_WIDTH * sizeof(double));
    }

    // gather final grid to rank 0
    double *flat_final = NULL;
    if (rank == 0)
        flat_final = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(double));

    MPI_Gatherv(flat, count_, MPI_DOUBLE,
                flat_final, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *fp = fopen("mpi.txt", "w");
        if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
        for (int i = 0; i < GRID_HEIGHT; i++) {
            for (int j = 0; j < GRID_WIDTH; j++) {
                fprintf(fp, "%8.8f ", flat_final[i*GRID_WIDTH + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        free(flat_final);
        free(counts);
        free(displs);
    }

    free(flat);
    free(u);
    free(next_u);
    MPI_Type_free(&row_t);
    MPI_Finalize();
    return 0;
}
