#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

// — Problem parameters —
#define GRID_HEIGHT 1500
#define GRID_WIDTH  1500
#define NUM_STEPS   2000

#define D    0.1   // diffusion coefficient
#define DT   1.0   // time step
#define DX   1.0   // Δx
#define DY   1.0   // Δy

int main(int argc, char **argv) {
    MPI_Init(&argc,&argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // 1) Split GRID_HEIGHT rows among 'size' ranks
    int base   = GRID_HEIGHT / size;
    int rem    = GRID_HEIGHT % size;
    int local  = base + (rank < rem ? 1 : 0);
    int start  = rank*base + (rank < rem ? rank : rem);

    // 2) Allocate local chunk + 2 ghost rows
    double (*u)[GRID_WIDTH]      = malloc((local+2)*GRID_WIDTH*sizeof(double));
    double (*next_u)[GRID_WIDTH] = malloc((local+2)*GRID_WIDTH*sizeof(double));
    if (!u || !next_u) MPI_Abort(MPI_COMM_WORLD,1);

    // 3) Initialize: zeros, then hot line in the global mid‐row
    for (int i = 0; i < local+2; i++)
        for (int j = 0; j < GRID_WIDTH; j++)
            u[i][j] = 0.0;
    int gmid = GRID_HEIGHT/2;
    if (gmid >= start && gmid < start+local) {
        int lmid = gmid - start + 1;  // +1 for top ghost
        for (int j = GRID_WIDTH/4; j < 3*GRID_WIDTH/4; j++)
            u[lmid][j] = 100.0;
    }
    memcpy(next_u, u, (local+2)*GRID_WIDTH*sizeof(double));

    // 4) Build MPI row type for ghost‐row exchange
    MPI_Datatype row_t;
    MPI_Type_contiguous(GRID_WIDTH, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // 5) Prepare for final gather: counts & displacements
    int *counts = NULL, *displs = NULL;
    if (rank==0) {
        counts = malloc(size*sizeof(int));
        displs = malloc(size*sizeof(int));
    }
    int my_count = local*GRID_WIDTH;
    MPI_Gather(&my_count,1,MPI_INT,
               counts,1,MPI_INT,
               0,MPI_COMM_WORLD);
    if (rank==0) {
        displs[0] = 0;
        for (int r=1; r<size; r++)
            displs[r] = displs[r-1] + counts[r-1];
    }

    // 6) Time‐stepping loop
    double t0 = MPI_Wtime();
    for (int step=1; step<=NUM_STEPS; step++) {
        // a) Exchange ghost rows
        int up   = (rank==0        ? MPI_PROC_NULL : rank-1);
        int down = (rank==size-1   ? MPI_PROC_NULL : rank+1);
        MPI_Sendrecv(u[1],     1, row_t, up,   0,
                     u[0],     1, row_t, up,   1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(u[local], 1, row_t, down, 1,
                     u[local+1],1,row_t, down,0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // b) Determine interior rows to update (Dirichlet top/bottom)
        int i0 = 1, i1 = local;
        if (start == 0)               i0 = 2;         // skip global top
        if (start+local == GRID_HEIGHT) i1 = local-1; // skip global bottom

        // c) OpenMP‐parallel stencil
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int i = i0; i <= i1; i++) {
                for (int j = 1; j < GRID_WIDTH-1; j++) {
                    double c    = u[i][j];
                    double lapx = (u[i][j+1] - 2*c + u[i][j-1])/(DX*DX);
                    double lapy = (u[i+1][j] - 2*c + u[i-1][j])/(DY*DY);
                    next_u[i][j] = c + DT * D * (lapx + lapy);
                }
            }
            #pragma omp single
            memcpy(u, next_u, (local+2)*GRID_WIDTH*sizeof(double));
        }
    }
    double t1 = MPI_Wtime();

    // 7) Gather final grid back to rank 0
    //    First flatten local real rows (skip ghost at 0 and local+1)
    double *my_flat = malloc(local*GRID_WIDTH*sizeof(double));
    for (int i=0; i<local; i++) {
        memcpy(my_flat + i*GRID_WIDTH,
               u[i+1],
               GRID_WIDTH*sizeof(double));
    }
    double *flat_final = NULL;
    if (rank==0)
        flat_final = malloc(GRID_HEIGHT*GRID_WIDTH*sizeof(double));

    MPI_Gatherv(my_flat, my_count, MPI_DOUBLE,
                flat_final, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 8) On rank 0, write the entire grid to "heat_output.txt"
    if (rank==0) {

        printf("Hybrid MPI+OpenMP (%d ranks × %d threads) → %.3f s\n",
               size, omp_get_max_threads(), t1 - t0);
        FILE *fp = fopen("hybrid.txt", "w");
        if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
        for (int i = 0; i < GRID_HEIGHT; i++) {
            for (int j = 0; j < GRID_WIDTH; j++) {
                fprintf(fp, "%8.8f ", flat_final[i*GRID_WIDTH + j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    // 9) Cleanup
    MPI_Type_free(&row_t);
    free(u);
    free(next_u);
    free(my_flat);
    free(counts);
    free(displs);
    free(flat_final);
    MPI_Finalize();
    return 0;
}
