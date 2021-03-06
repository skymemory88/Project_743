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
    if (argc != 6)
        throw runtime_error("Incorrect argument number! 1. x_size, 2. y_size, 3. z_size, 4. Iteration limit 5. Normalized coupling strength.\n");
    const int local_xsize = atoi(argv[1]); //set local lattice size in x direction
    const int local_ysize = atoi(argv[2]); //set local lattice size in y direction
    //const int local_zsize = atoi(argv[3]); //set local lattice size in z direction
    const int limit = atoi(argv[4]); //set the limit of how may rounds the simulation can evolve
    const int halo = 1;              //set halo size for the local lattice
    clock_t t_start = clock();       //bench mark time point

    lattice<signed char, LatticeForm::square> grid(local_xsize + 2 * halo, local_ysize + 2 * halo);
    //initialize a local lattice with halo boarder, options: "square", "kagome", "triangular", "circular"
    printf("Local grid size: %d x %d.\n", local_xsize, local_ysize);

    const float K = atof(argv[5]);                        //K contains info regarding coupling strength to thermal fluctuation ratio
    const double epsilon = 2.0 * sqrt(0.5); //define toloerance as the smallest energy difference can be produced, other than zero, by flipping one spin
    double E_site = 0.0;                        //declare local energy
    double E_old = 0.0;                         //declare energy before updates
    double E_new = 0.0;                         //declare energy after upHdates
    auto new_grid = grid;                       //duplicate the current grid for updating
    int omp_size = omp_get_max_threads();   //get the total thread number
    int round = 1;                         //parameter to keep track of the iteration cycles
    ofstream fout;                         //declare an filestream
    fout.precision();                      //set the precision of data to be written
    fout.open("Global_energy.dat");        //open the file and give a file name

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
                grid(i, j) = static_cast<signed char>((Rand() > 0.5) ? -1 : 1); //randomly assign the spin state on each site
            }
        } //randomly assign spin values to the lattice sites

#pragma omp for reduction(+ : E_new) schedule(auto)                     //calculate the total energy of the configuration
        for (int i = halo; i < grid.xsize - halo; i++) //avoid double counting
        {
            for (int j = halo; j < grid.ysize - halo; j++)
            {
                E_new += -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i, j + 1));
            }
        } //using the initialized grid to compute the global energy

        ///////////////////////////////start updating algorithm//////////////////////////////
        
#pragma omp master
        {
            new_grid = grid; //duplicate the grid for updating sequence
            cout << "Initial energy: " << E_new << endl;
            printf("Total local thread number: %d.\n", omp_size);
            grid.map("initial.dat", 0);
        }

        do
        { //continue the algorithm until a stable state
#pragma omp master
            {
                E_old = E_new; //pass on the new global energy
                E_new = 0.0; //reset the global energy for next update
                //printf("start round %d.\n", round); //checkpoint for debugging
            }
#pragma omp for schedule(auto)
            for (int i = halo; i < grid.xsize - halo; i++)
            {
                for (int j = halo; j < grid.ysize - halo; j++)
                {
                    E_site = -1.0 * grid(i, j) * (grid(i + 1, j) + grid(i - 1, j) + grid(i, j + 1) + grid(i, j - 1)) + -1.0 * sqrt(0.5) * grid(i, j) * (grid(i + 1, j + 1) + grid(i - 1, j - 1) + grid(i - 1, j + 1) + grid(i + 1, j - 1));
                    if (E_site > 0) //can be replaced with explicit "E_init > E_fin" conditions
                    {
                        new_grid(i, j) = -grid(i, j);
                        //printf("Spin flipped! case 1\n");  //checkpoint
                    }
                    else if (E_site < 0)
                    {
                        new_grid(i, j) = (Rand() >= exp(2.0 * K * E_site) ? grid(i, j) : -grid(i,j));
                        //if(omp_rank == 0)
                        //  printf("Spin flipped! case 3. Probability = %.4f.\n", exp(2.0 * E_site)); //checkpoint
                    }
                    else
                    {
                        new_grid(i, j) = grid(i, j);
                    }
                    //if(omp_rank == 0)
                        //printf("Local energy = %.4e.\n", E_site); //checkpoint
                }
            }  //Metropolis Algorithm to update the spin configuration on the new grid
#pragma omp for reduction(+ : E_new) schedule(auto)                        //calculate the total energy of the configuration
            for (int i = halo; i < new_grid.xsize - halo; i++) //avoid double counting
            {
                for (int j = halo; j < new_grid.ysize - halo; j++)
                {
                    E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
                }
            }
#pragma omp master
            {
                if (round % (limit / 50) == 0) //report to screen every 100 round of evolution
                {
                    fout << round << "\t" << E_new << endl;
                    printf("Round %d finished. Current E = %.2f, last E = %.2f, difference = %.5f.\n", round, E_new, E_old, std::abs(E_new - E_old));
                }
                grid = new_grid;                             //update the current grid to the new one
                round++;
            }
        } while (std::abs(E_new - E_old) > epsilon && round < limit);
#ifdef _OPENMP
    }
#endif

fout.close();

    if (std::abs(E_new - E_old) < epsilon)
    {
        printf("Energy converged, landscape mapped! total iteration: %d \n", round);
        grid.map("spin_map.dat", 0);
    }
    else if (round >= limit) //stop the program if it doesn't converge
        printf("Evolution round exceeded the limit (%d rounds), simulation terminated and current spin configuration exported.\n", limit);

    printf("Time used: %.2f seconds. \n", (float)((clock() - t_start) / CLOCKS_PER_SEC)); //print out total time lapsed
    return 0;
}