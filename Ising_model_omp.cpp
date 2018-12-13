#include "Lattice.hpp"
#include "mtrand.hpp"
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <ctime>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 5)
        throw runtime_error("Incorrect argument number! 1. x_size, 2. y_size, 3. z_size, 4. Round limit.");
    const int local_xsize = atoi(argv[1]); //set local lattice size in x direction
    const int local_ysize = atoi(argv[2]); //set local lattice size in y direction
    const int local_zsize = atoi(argv[3]); //set local lattice size in z direction
    const int limit = atoi(argv[4]); //set the limit of how may rounds the simulation can evolve
    const int halo = 1;              //set halo size for the local lattice
    clock_t t_start = clock();       //bench mark time point

    lattice<signed char, LatticeForm::square> grid(local_xsize + 2 * halo, local_ysize + 2 * halo);
    //initialize a local lattice with halo boarder, options: "square", "kagome", "triangular", "circular"
    printf("Local grid size: %d x %d.\n", local_xsize, local_ysize);

    const float K = 0.1;                        //K contains info regarding coupling strength to thermal fluctuation ratio
    const double epsilon = 4.0 * (1.0 + sqrt(0.5)); //define toloerance
    double E_site = 0.0;                        //declare local energy
    double E_old = 0.0;                         //declare energy before updates
    double E_new = 0.0;                         //declare energy after upHdates
    auto new_grid = grid;                       //duplicate the current grid for updating
    int omp_size = omp_get_max_threads();   //get the total thread number
    int round = 1;                         //parameter to keep track of the iteration cycles

    ///////////////////////////////Initialize the lattice///////////////////////////////////

#ifdef _OPENMP
#pragma omp parallel shared(E_old, E_new, new_grid, grid, round) private(E_site)
    {
        int omp_rank = omp_get_thread_num();
        mtrand Rand(omp_rank);
#pragma omp for schedule(auto) //optional schedule(guided)
#endif
        for (int i = halo; i < grid.xsize - halo; i++) //assign values to the grid
        {
            for (int j = halo; j < grid.ysize - halo; j++)
            {
                grid(i, j) = static_cast<signed char>((Rand() > 0.5) ? 1 : -1); //randomly assign the spin state on each site
            }
        } //randomly assign spin values to the lattice sites
        new_grid = grid; //duplicate the grid for updating sequence

#pragma omp for reduction(+ : E_new)                     //calculate the total energy of the configuration
        for (int i = halo; i < new_grid.xsize - halo; i++) //avoid double counting
        {
            for (int j = halo; j < new_grid.ysize - halo; j++)
            {
                E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
            }
        }

        ///////////////////////////////start updating algorithm//////////////////////////////
        
        if (omp_rank == 0)
        {
            cout << "Initial energy: " << E_new << endl;
            printf("Total local thread number: %d.\n", omp_size);
        }

#pragma omp master
        {
            grid.map("initial.dat", 0);
            ofstream fout;
            fout.precision();
            fout.open("Global_energy.dat");
        }
        do
        { //continue the algorithm until a stable state
            E_old = E_new;
            E_new = 0;
#pragma omp for schedule(auto)
            for (int i = halo; i < grid.xsize - halo; i++)
            {
                for (int j = halo; j < grid.ysize - halo; j++)
                {
                    E_site = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_site >= 0) //can be replaced with explicit "E_init > E_fin" conditions
                    {
                        new_grid(i, j) = -grid(i, j);
                        //printf("Spin flipped! case 1\n");  //checkpoint
                    }
                    else if (Rand() < exp(2.0 * K * E_site))
                    {
                        new_grid(i, j) = -grid(i, j);
                        //if(omp_rank == 0)
                        //  printf("Spin flipped! case 3. Probability = %.4f.\n", exp(2.0 * E_site)); //checkpoint
                    }
                    //printf("Local energy = %.4e.\n", E_site); //checkpoint
                }
            }
            //Metropolis Algorithm for the inner grid that doesn't rely on the halos
            grid = new_grid; //update the current grid to the new one

#pragma omp for reduction(+ : E_new)                         //calculate the total energy of the configuration
            for (int i = halo; i < new_grid.xsize - halo; i++) //avoid double counting
            {
                for (int j = halo; j < new_grid.ysize - halo; j++)
                {
                    E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
                }
            }
#pragma omp master
            {
                if (round % (limit / 100) == 0) //report to screen every 100 round of evolution
                    fout << round << "\t" << E_new << endl;
                    printf("Round %d finished. Current E = %.2f, last E = %.2f, difference = %.5f.\n", round, E_new, E_old, std::abs(E_new - E_old));
                round++;
            }
        } while (std::abs(E_new - E_old) > epsilon && round < limit);
#ifdef _OPENMP
    }
#endif

fout.close();

    if (std::abs(E_new - E_old) < epsilon)
    {
        printf("Energy converged, landscape mapped!\n");
        grid.map("spin_map.dat", 0);
    }
    else if (round >= limit) //stop the program if it doesn't converge
        printf("Evolution round exceeded the limit (%d rounds), simulation terminated and current spin configuration exported.\n", limit);

    printf("Time used: %.2f seconds. \n", (float)((clock() - t_start) / CLOCKS_PER_SEC)); //print out total time lapsed
    return 0;
}