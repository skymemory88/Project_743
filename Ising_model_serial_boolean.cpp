#include "Lattice.hpp"
#include "mtrand.hpp"
#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

int main(int, char **)
{
    mtrand Rand(time(0));
    const int limit = 2000; //set the limit of how may rounds the simulation can evolve
    //const int dimension = 2;     //set space dimension, option: 1,2,3
    int local_xsize = 1000; //set local lattice size in x direction
    int local_ysize = 1000; //set local lattice size in y direction
    int halo = 1;          //set halo size for the local lattice
    //int local_zsize = 100; //set local lattice size in z direction

     clock_t t_start = clock(); //bench mark time point

    lattice<bool> grid(local_xsize + 2 * halo, local_ysize + 2 * halo);
    //initialize a local lattice with halo boarder, options: "square", "kagome", "triangular", "circular"
    printf("Grid size: %d x %d. Halo size: %d.\n", grid.xsize, grid.ysize, halo);
    grid.map("initial", 0);

    const float K = 0.1;                        //K contains info regarding coupling strength to thermal fluctuation ratio
    const double epsilon = 4 * (1 + sqrt(0.5)); //define toloerance
    double E_init = 0.0;           //declare local energy
    double E_old = 0.0;            //declare energy before updates
    double E_new = 0.0;            //declare energy after upHdates
    int round = 1;                 //parameter to keep track of the iteration cycles

    ///////////////////////////////Initialize the lattice///////////////////////////////////

    for (int i = halo; i < grid.xsize - halo; i++) //assign values to the grid
    {
        for (int j = halo; j < grid.ysize - halo; j++)
        {
            grid(i, j) = ((Rand() > 0.5) ? true : false); //randomly assign the spin state on each site
        }
    } //randomly assign spin values to the lattice sites

    lattice<bool> new_grid = grid; //duplicate the current grid for updating

    for (int i = halo; i < grid.xsize - halo; i++)
    {
        for (int j = halo; j < grid.ysize - halo; j++)
        {
            E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
        }
    } //calculate the total energy of the initial configuration

    ///////////////////////////////start updating algorithm//////////////////////////////

    do
    { //continue the algorithm until a stable state
        E_old = E_new;
        E_new = 0; //reset the total new energy
        for (int i = halo; i < grid.xsize - halo; i++)
        {
            for (int j = halo; j < grid.ysize - halo; j++)
            {
                E_init = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                if (E_init > 0) //can be replaced with explicit "E_init > E_fin" conditions
                {
                    new_grid(i, j) = -grid(i, j);
                    //printf("Spin flipped! case 1\n");  //checkpoint
                }
                else if (E_init == 0 && Rand() > 0.5)
                {
                    new_grid(i, j) = static_cast<char>((Rand(1) > 0.5) ? 1 : -1);
                    // printf("Spin flipped! 50-50!\n");  //checkpoint
                }
                else if (E_init < 0 && Rand() < exp(2.0 * K * E_init))
                {
                    new_grid(i, j) = -grid(i, j);
                    //if(omp_rank == 0)
                    //  printf("Spin flipped! case 3. Probability = %.4f.\n", exp(2.0 * E_init)); //checkpoint
                }
                //printf("Local energy = %.4e.\n", E_init); //checkpoint
            }
        }
        //Metropolis Algorithm to update the spin configuration on the new grid

        for (int i = halo; i < grid.xsize - halo; i++) //avoid double counting
        {
            for (int j = halo; j < grid.ysize - halo; j++)
            {
                E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
            }
        } //calculate the total energy of the new configuration
        // printf("E(next round) = %.4e.\n", E_new); //checkpoint

        if (round % (limit / 100) == 0) //report to screen every 100 round of evolution
            printf("Round %d finished. Current E = %.4f.\n", round, E_new);

        grid = new_grid; //duplicate the current grid for updating
        round++;
    } while (std::abs(E_new - E_old) > epsilon && round < limit);

    if (std::abs(E_new - E_old) < epsilon)
    {
        printf("Energy converged, landscape mapped!\n");
        grid.map("spin_map", 0);
    }
    else if (round > limit) //stop the program if it doesn't converge
        printf("Evolution round exceeded the limit (%d rounds), simulation terminated and current spin configuration exported.\n", limit);

 printf("Time used: %.2f seconds. \n", (float)((clock() - t_start)/CLOCKS_PER_SEC) ); //print out total time lapsed

    return 0;
}