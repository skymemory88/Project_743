#include "Lattice.hpp"
#include "mtrand.hpp"
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <array>

using namespace std;

int main(int, char **)
{
    const int limit = 2000;       //set the limit of how may rounds the simulation can evolve
    const int dimension = 2;      //set space dimension, option: 1,2,3
    const int local_xsize = 1000; //set local lattice size in x direction
    const int local_ysize = 1000; //set local lattice size in y direction
    const int halo = 1;           //set halo size for the local lattice
    //int local_zsize = 100; //set local lattice size in z direction

    /////////////////////Initialize global communication and MPI topology//////////////////
    int proc_num;
    //int mpi_support;
    //MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &mpi_support);
    MPI_Init(NULL, NULL);
    //if(mpi_support != MPI_THREAD_FUNNELD)
    //  printf("Unmatched MPI_Thread support detected, provided: %d", mpi_support);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num); //get the number of available nodes

    int pDims[dimension] = {0}; //array contains processors per dimension
    int period[dimension] = {0};
    MPI_Dims_create(proc_num, dimension, pDims); //let MPI decides processor-per-dimension

    MPI_Comm cartcomm;                                                       //new communicator for MPI virtual topology
    MPI_Cart_create(MPI_COMM_WORLD, dimension, pDims, period, 1, &cartcomm); //initiate MPI virtual topology

    int t_start = MPI_Wtime();
    int root = 0;
    int myrank;
    int mpi_index;           //index used by MPI_Wait function
    int mycoords[dimension]; //define processor coordinate
    MPI_Comm_rank(cartcomm, &myrank);
    MPI_Cart_coords(cartcomm, myrank, dimension, &mycoords[0]);
    int myrank_x = mycoords[0];
    int myrank_y = mycoords[1];
    printf("Global position: (%d, %d).\n", mycoords[0], mycoords[1]); //checkpoint
    //int myrank_z = mycoords[2];
    int global_xsize = local_xsize * pDims[0];
    int global_ysize = local_ysize * pDims[1];
    //int global_zsize = local_zsize * pDims[2]; //compute the size of the global lattice

    mtrand Rand(MPI_Wtime()); //seed the random number generator

    MPI_Request send[8];      //send request for all Nearest Neighbours and Next Nearest Neighbours
    MPI_Request lrecv[4];     //receive request for all Nearest Neighbours Neighbours
    MPI_Request crecv[4];     //receive request for all Next Nearest Neighbours
    MPI_Status lrecv_stat[4]; //status associate with edge requests.
    MPI_Status crecv_stat[4]; //status associate with corner requests.

    int NORTH, SOUTH, WEST, EAST; //declare global nearest neighbours
    int NE, NW, SE, SW;           //declare global next nearest neighbours

    int north = Rand(100);
    int south = north + 5;
    int west = south + 5;
    int east = west + 5;
    //define tags for global nearest neighbours

    int northwest = west + 5;
    int northeast = northwest + 5;
    int southwest = northeast + 5;
    int southeast = southwest + 5; //define tags for global next nearest neighbours

    MPI_Cart_shift(cartcomm, 0, 1, &WEST, &EAST);   //get the ranks of row neighbours
    MPI_Cart_shift(cartcomm, 1, 1, &SOUTH, &NORTH); //get the ranks of column neighbours

    MPI_Isend(&NORTH, 1, MPI_CHAR, WEST, northeast, cartcomm, &send[0]);
    MPI_Isend(&NORTH, 1, MPI_CHAR, EAST, northwest, cartcomm, &send[1]);
    MPI_Isend(&SOUTH, 1, MPI_CHAR, WEST, southeast, cartcomm, &send[2]);
    MPI_Isend(&SOUTH, 1, MPI_CHAR, EAST, southwest, cartcomm, &send[3]);
    //send the coordinates of the next nearest neighbours to each other

    MPI_Recv(&NW, 1, MPI_CHAR, WEST, northwest, cartcomm, &crecv_stat[0]);
    MPI_Recv(&NE, 1, MPI_CHAR, EAST, northeast, cartcomm, &crecv_stat[1]);
    MPI_Recv(&SW, 1, MPI_CHAR, WEST, southeast, cartcomm, &crecv_stat[2]);
    MPI_Recv(&SE, 1, MPI_CHAR, EAST, southwest, cartcomm, &crecv_stat[3]);
    //receive the coordinates of the next nearest neighbours

    lattice<char> grid(local_xsize + 2 * halo, local_ysize + 2 * halo);
    //initialize a local lattice with halo boarder, options: "square", "kagome", "triangular", "circular"
    printf("Local grid size: %d x %d. Halo size: %d.\n", grid.xsize, grid.ysize, halo);
    printf("Global grid size: %d x %d.\n", global_xsize, global_ysize);
    //grid.map("local_initial", 0);

    const float K = 0.1;                  //K contains info regarding coupling strength to thermal fluctuation ratio
    double epsilon = 4 * (1 + sqrt(0.5)); //define toloerance
    double E_init = 0.0;                  //declare local energy for updating
    double local_E_i = 0.0;               //declare energy before updates
    double local_E_f = 0.0;               //declare energy after updates
    double global_E_i = 0.0;              //declare global energy beofre update (grid)
    double global_E_f = 0.0;              //declare global energy after update (new_grid)
    int round = 1;                        //parameter to keep track of the iteration cycles
    lattice<char> new_grid = grid;        //duplicate the current grid for updating

    ///////////////////////////////Initialize the lattice///////////////////////////////////
    if (myrank == root)
        printf("Total node number: %d.\n", proc_num);

    for (int i = halo; i < (grid.xsize - halo); i++) //assign values to the grid
    {
        for (int j = halo; j < (grid.ysize - halo); j++)
        {
            grid(i, j) = (char)((Rand() > 0.5) ? -1 : 1); //randomly assign the spin state on each site
        }
    } //randomly assign spin values to the lattice sites

    MPI_Datatype column_halo;
    MPI_Type_vector(local_ysize, 1, grid.xsize, MPI_CHAR, &column_halo);
    MPI_Type_commit(&column_halo); //declare column halo to pass to neighbours

    MPI_Isend(&grid(halo, halo), local_xsize, MPI_CHAR, NORTH, south, cartcomm, &send[0]);
    MPI_Isend(&grid(grid.xsize - halo, halo), local_ysize, MPI_CHAR, SOUTH, north, cartcomm, &send[1]);
    MPI_Isend(&grid(grid.xsize - halo, halo), 1, column_halo, WEST, east, cartcomm, &send[2]);
    MPI_Isend(&grid(halo, halo), 1, column_halo, EAST, west, cartcomm, &send[3]);
    //non-blocking send edge halos

    MPI_Isend(&grid(grid.xsize - halo, grid.ysize - halo), 1, MPI_CHAR, SE, northwest, cartcomm, &send[4]);
    MPI_Isend(&grid(halo, grid.ysize - halo), 1, MPI_CHAR, NE, southwest, cartcomm, &send[5]);
    MPI_Isend(&grid(grid.ysize - halo, halo), 1, MPI_CHAR, SW, northeast, cartcomm, &send[6]);
    MPI_Isend(&grid(halo, halo), 1, MPI_CHAR, NW, southeast, cartcomm, &send[7]);
    //non-blocking send corner halos

    MPI_Irecv(&grid(halo, 0), local_xsize, MPI_CHAR, NORTH, north, cartcomm, &lrecv[0]);
    MPI_Irecv(&grid(halo, grid.ysize), local_xsize, MPI_CHAR, SOUTH, south, cartcomm, &lrecv[1]);
    MPI_Irecv(&grid(halo, halo), 1, column_halo, WEST, west, cartcomm, &lrecv[2]);
    MPI_Irecv(&grid(grid.xsize - halo, halo), 1, column_halo, EAST, east, cartcomm, &lrecv[3]);
    //non-blocking request for edge halos

    MPI_Irecv(&grid(0, 0), 1, MPI_CHAR, NW, northwest, cartcomm, &crecv[0]);
    MPI_Irecv(&grid(0, grid.ysize), 1, MPI_CHAR, NE, northeast, cartcomm, &crecv[1]);
    MPI_Irecv(&grid(grid.xsize, 0), 1, MPI_CHAR, SW, southwest, cartcomm, &crecv[2]);
    MPI_Irecv(&grid(grid.xsize, grid.ysize), 1, MPI_CHAR, SE, southeast, cartcomm, &crecv[3]);
    //non-blocking request for corner halos

    MPI_Waitall(4, lrecv, lrecv_stat); //wait untill all line halos received
    MPI_Waitall(4, crecv, crecv_stat); //wait untill all corner halos received

  for (int i = halo; i < (grid.xsize - halo); i++)
    {
        for (int j = halo; j < (grid.ysize - halo); j++)
        {
            local_E_i += -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i, j + 1)); //avoid double counting
        }
    }
   
    MPI_Allreduce(&global_E_i, &local_E_i, 1, MPI_DOUBLE, MPI_SUM, cartcomm); //compute the global energy of the current spin configuration over all nodes
    global_E_f = global_E_i;
 
    ///////////////////////////////start updating algorithm//////////////////////////////

    do
    {
        global_E_i = global_E_f; //update the global total energy of last step to the next;
        global_E_f = 0;          //reset the global total energy of the next step
        local_E_i = 0;           //reset the total energy of the local node
        local_E_f = 0;           //reset the total energy of the local node

        MPI_Isend(&grid(halo, halo), local_xsize, MPI_CHAR, NORTH, south, cartcomm, &send[0]);
        MPI_Isend(&grid(grid.xsize - halo, halo), local_ysize, MPI_CHAR, SOUTH, north, cartcomm, &send[1]);
        MPI_Isend(&grid(grid.xsize - halo, halo), 1, column_halo, WEST, east, cartcomm, &send[2]);
        MPI_Isend(&grid(halo, halo), 1, column_halo, EAST, west, cartcomm, &send[3]);
        //non-blocking send edge halos

        MPI_Isend(&grid(grid.xsize - halo, grid.ysize - halo), 1, MPI_CHAR, SE, northwest, cartcomm, &send[4]);
        MPI_Isend(&grid(halo, grid.ysize - halo), 1, MPI_CHAR, NE, southwest, cartcomm, &send[5]);
        MPI_Isend(&grid(grid.ysize - halo, halo), 1, MPI_CHAR, SW, northeast, cartcomm, &send[6]);
        MPI_Isend(&grid(halo, halo), 1, MPI_CHAR, NW, southeast, cartcomm, &send[7]);
        //non-blocking send corner halos

        MPI_Irecv(&grid(halo, 0), local_xsize, MPI_CHAR, NORTH, north, cartcomm, &lrecv[0]);
        MPI_Irecv(&grid(halo, grid.ysize), local_xsize, MPI_CHAR, SOUTH, south, cartcomm, &lrecv[1]);
        MPI_Irecv(&grid(halo, halo), 1, column_halo, WEST, west, cartcomm, &lrecv[2]);
        MPI_Irecv(&grid(grid.xsize - halo, halo), 1, column_halo, EAST, east, cartcomm, &lrecv[3]);
        //non-blocking request for edge halos

        MPI_Irecv(&grid(0, 0), 1, MPI_CHAR, NW, northwest, cartcomm, &crecv[0]);
        MPI_Irecv(&grid(0, grid.ysize), 1, MPI_CHAR, NE, northeast, cartcomm, &crecv[1]);
        MPI_Irecv(&grid(grid.xsize, 0), 1, MPI_CHAR, SW, southwest, cartcomm, &crecv[2]);
        MPI_Irecv(&grid(grid.xsize, grid.ysize), 1, MPI_CHAR, SE, southeast, cartcomm, &crecv[3]);
        //non-blocking request for corner halos

      for (int i = 2 * halo; i < (grid.xsize - 2 * halo); i++)
        {
            for (int j = 2 * halo; j < (grid.ysize - 2 * halo); j++)
            {
                E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                    new_grid(i, j) = -grid(i, j);
                else if (E_init == 0 && Rand() > 0.5)
                    new_grid(i, j) = (char)((Rand(1) > 0.5) ? 1 : -1);
                else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                    new_grid(i, j) = -grid(i, j);
            }
        }

        //Metropolis Algorithm for the inner grid that doesn't rely on the halos

        for (int loop = 0; loop < 4; loop++)
        {
            int i, j;
            MPI_Waitany(4, lrecv, &mpi_index, lrecv_stat); //wait for edge halos
            switch (mpi_index)
            {
            case 0: //received north halo
                j = halo;
                for (i = halo; i < grid.xsize - halo; i++)
                {
                    E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                        new_grid(i, j) = -grid(i, j);
                    else if (E_init == 0 && Rand() > 0.5)
                        new_grid(i, j) = ((Rand(1) > 0.5) ? 1 : -1);
                    else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                        new_grid(i, j) = -grid(i, j);
                }
                j = 0;

            case 1: //received south halo
                j = grid.ysize - halo;
                for (i = halo; i < grid.xsize - halo; i++)
                {
                    E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                        new_grid(i, j) = -grid(i, j);
                    else if (E_init == 0 && Rand() > 0.5)
                        new_grid(i, j) = (char)((Rand(1) > 0.5) ? 1 : -1);
                    else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                        new_grid(i, j) = -grid(i, j);
                }
                j = 0;

            case 2: //received west halo
                i = halo;
                for (j = halo; j < grid.ysize - halo; j++)
                {
                    E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                        new_grid(i, j) = -grid(i, j);
                    else if (E_init == 0 && Rand() > 0.5)
                        new_grid(i, j) = (char)((Rand(1) > 0.5) ? 1 : -1);
                    else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                        new_grid(i, j) = -grid(i, j);
                }
                i = 0;

            case 3: //received east halo
                i = grid.xsize - halo;
                for (j = halo; j < grid.xsize - halo; j++)
                {
                    E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                        new_grid(i, j) = -grid(i, j);
                    else if (E_init == 0 && Rand() > 0.5)
                        new_grid(i, j) = (char)((Rand(1) > 0.5) ? 1 : -1);
                    else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                        new_grid(i, j) = -grid(i, j);
                }
                i = 0;
            }
        }
        //Metropolis Algorithm for the boarders that rely on the halos

        for (int loop = 0; loop < 4; loop++)
        {
            int i, j;
            MPI_Waitany(4, crecv, &mpi_index, crecv_stat);
            switch (mpi_index)
            {
            case 0: //received nw halo
                i = halo;
                j = halo;
            case 1: //received sw halo
                i = grid.xsize - halo;
                j = halo;
            case 2: //received ne halo
                i = halo;
                j = grid.ysize - halo;
            case 3: //received se halo
                i = grid.xsize - halo;
                j = grid.ysize - halo;
            }

            E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
            if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                new_grid(i, j) = -grid(i, j);
            else if (E_init == 0 && Rand() > 0.5)
                new_grid(i, j) = (char)((Rand(1) > 0.5) ? 1 : -1);
            else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                new_grid(i, j) = -grid(i, j);

            i = 0;
            j = 0; //get rid of the temporary iterator
        }
        //Metropolis algorithm on the corners

       for (int i = halo; i < (grid.xsize - halo); i++)
        {
            for (int j = halo; j < (grid.ysize - halo); j++)
            {
                local_E_f += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1)); //avoid double counting
            }
        }

        //MPI_Reduce(&global_E_i, &local_E_i, 1, MPI_DOUBLE, MPI_SUM, root, cartcomm); //compute the global energy of the previous step on the root node
        //MPI_Reduce(&global_E_f, &local_E_f, 1, MPI_DOUBLE, MPI_SUM, root, cartcomm); //compute the global energy of the current step on the root node

        //MPI_Allreduce(&global_E_i, &local_E_i, 1, MPI_DOUBLE, MPI_SUM, cartcomm); //compute the global energy on all nodes
        MPI_Allreduce(&global_E_f, &local_E_f, 1, MPI_DOUBLE, MPI_SUM, cartcomm); //compute the global energy on all nodes

        if (myrank == root)
            if (round % (limit / 100) == 0) //report to screen every 100 round of evolution
                printf("Round %d finished. Current E = %.4f.\n", round, global_E_f);

        grid = new_grid; //duplicate the current grid for updating
        round++;
    } while (std::abs(global_E_f - global_E_i) > epsilon && round < limit);

    if (myrank == root && abs(global_E_f - global_E_i) < epsilon)
        printf("Energy converged, landscape mapped!\n");
    else if (round > limit) //stop the program if it doesn't converge
        printf("Evolution round exceeded the limit (%d rounds), simulation terminated and current spin configuration exported.\n", limit);

    /////////////////////////////////////////////Output the final results///////////////////////////////////
    int mpi_of;
    MPI_Offset disp = myrank_x * local_xsize + myrank_y * local_xsize * local_ysize * pDims[0];
    MPI_File gmap;
    char *buffer;
    buffer = (char *)malloc(grid.val.size() * sizeof(char));
    mpi_of = MPI_File_open(cartcomm, "Global_spin_map.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &gmap);
    //mpi_of = MPI_File_open(cartcomm, "Global_spin_map", MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_SEQUENTIAL | MPI_MODE_APPEND, MPI_INFO_NULL, &gmap); //mpi sequential writing

    if (mpi_of)
        printf("Unable to output global map!\n");
    else
    {
        MPI_File_set_view(gmap, disp * sizeof(char), MPI_CHAR, MPI_CHAR, "Global_spin_map.dat", MPI_INFO_NULL);
        MPI_File_write_at_all(gmap, disp, buffer, local_xsize * local_ysize, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&gmap);
    }

    printf("Total time used: %.2f seconds.\n", MPI_Wtime() - t_start);
    MPI_Finalize();
    return 0;
}