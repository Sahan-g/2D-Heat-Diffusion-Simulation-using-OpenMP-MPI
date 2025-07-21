# 2D Heat Diffusion Simulation

An explicit five-point stencil solver for the 2D heat diffusion equation on a large grid, implemented in four variants:

1. **serial**: Single-threaded C implementation
2. **openmp**: Shared-memory version using OpenMP threads
3. **mpi**: Distributed-memory version using MPI (one process per rank)
4. **hybrid**: Mixed MPI+OpenMP version (one rank per process, multiple threads per rank)

Each variant writes its final temperature field to a text file in row-major order with eight decimal places.

## Prerequisites

### Compiler & Libraries
- GCC (or any C99-compatible compiler)
- OpenMP support (`-fopenmp`)
- MPI library (e.g., MPICH or OpenMPI)
- Python 3
- NumPy (required for `error.py`)

## Building

To compile all four executables:

```bash
# Serial
gcc -O3 -o serial serial.c

# OpenMP
gcc -O3 -fopenmp -o openmp openmp.c

# MPI
mpicc -O3 -o mpi mpi.c

# Hybrid
mpicc -O3 -fopenmp -o hybrid hybrid.c
```
### Running
Serial 
```bash
./serial
```
## OpenMp

```bash
export OMP_NUM_THREADS=4
./openmp
```

## MPI 
```bash
mpirun -np 4 ./mpi
```

## Hybrid 
```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./hybrid
```
### Output

Each version writes the final temperature field to:

serial.txt

openmp.txt

mpi.txt

hybrid.txt

Files contain floating-point values in row-major order with 8 decimal places.

## Accuracy Verification

```bash
python3 error.py
```



