<<<<<<< HEAD
#include <mpi.h>
#include <iostream>

int main(int, char**)
{
    int proc_num;
    int dimension = 2;
    int myrank_init;
    int pDims[dimension] = {0};

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num)
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_init);
    printf("Total process number: %d. \n", proc_num);
    printf("Current rank: %d.\n", myrank_init)
    
    MPI_Comm cartcomm;
    MPI_Dims_create(proc_num, dimension, pDims);
    MPI_Cart_create(MPI_COMM_WORLD, dimension, pDims, period, 1, &cartcomm);

    int myrank;
    int mycoords[dimension];

    MPI_Comm_rank(cartcomm, &myrank);
    MPI_Cart_coords(cartcomm, myrank, dimension, &mycoords[0]);
    
    return 0;
=======
#include <mpi.h>
#include <iostream>

int main(int, char**)
{
    int proc_num;
    int dimension = 2;
    int myrank_init;
    int pDims[dimension] = {};
    int period[dimension] = {};

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_init);
    
    printf("Total process number: %d. \n", proc_num);
    printf("Current rank: %d.\n", myrank_init);

    int myrank;
    int mycoords[dimension];

    MPI_Comm cartcomm;
    MPI_Dims_create(proc_num, dimension, pDims);
    MPI_Cart_create(MPI_COMM_WORLD, dimension, pDims, period, 0, &cartcomm);

    MPI_Comm_rank(cartcomm, &myrank);
    MPI_Cart_coords(cartcomm, myrank, dimension, mycoords);
    
    printf("Previous rank: %d, new rank under virtual topology: %d.\n", myrank_init, myrank);

    if (myrank == 0)
        printf("Processor dimensions: %d x %d.\n", pDims[0], pDims[1]);

    return 0;
>>>>>>> 0e92ca28349d18c01d2f52999f6b2ec8c95a58a8
}