#include "Lattice.hpp"
#include "mtrand.hpp"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <ctime>

using namespace std;

int main(int argc, char **argv)
{
    if (argc != 6)
        throw runtime_error("Incorrect argument number! 1. x_size, 2. y_size, 3. z_size, 4. Iteration limit, 5. Normalized coupling strength");
    mtrand Rand(time(0));
    //const int dimension = 2;     //set space dimension, option: 1,2,3
    int local_xsize = atoi(argv[1]); //set local lattice size in x direction
    int local_ysize = atoi(argv[2]); //set local lattice size in y direction
    //int local_zsize = atoi(argv[3]); //set local lattice size in z direction   
    const int limit = atoi(argv[4]); //set the limit of how may rounds the simulation can evolve
    int halo = 1;          //set halo size for the local lattice
    clock_t t_start = clock(); //bench mark time point

    lattice<signed char, LatticeForm::square> grid(local_xsize + 2 * halo, local_ysize + 2 * halo);
    //initialize a local lattice with halo boarder, options: "square", "kagome", "triangular", "circular"
    printf("Grid size: %d x %d. Halo size: %d.\n", grid.xsize, grid.ysize, halo);

    const double K = atof(argv[5]);     //K contains info regarding coupling strength to thermal fluctuation ratio
    const double epsilon = 2.0 * sqrt(0.5); //define toloerance as the smallest energy difference can be produced, other than zero, by flipping one spin
    double E_site = 0.0;           //declare local energy
    double E_old = 0.0;            //declare energy before updates
    double E_new = 0.0;            //declare energy after upHdates
    int round = 1;                 //parameter to keep track of the iteration cycles

    ///////////////////////////////Initialize the lattice///////////////////////////////////

    for (int i = halo; i < grid.xsize - halo; i++) //assign values to the grid
    {
        for (int j = halo; j < grid.ysize - halo; j++)
        {
            grid(i, j) = static_cast<signed char>((Rand() > 0.5) ? -1 : 1); //randomly assign the spin state on each site
            //cout << "Local spin: " << grid(i,j) << endl;
        }
    } //randomly assign spin values to the lattice sites
    
    grid.map("initial.dat", 0);
    auto new_grid = grid; //duplicate the current grid for updating

    for (int i = halo; i < new_grid.xsize - halo; i++)
    {
        for (int j = halo; j < new_grid.ysize - halo; j++)
        {
            E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1)); //avoid doule counting
        }
    } //calculate the total energy of the initial configuration
    cout << "Initial energy: " << E_new << endl;
   
    ///////////////////////////////start updating algorithm//////////////////////////////
   
    ofstream fout;
    fout.precision(6);
    fout.open("Global_energy.dat");

    do
    { //continue the algorithm until a stable state
        E_old = E_new;
        E_new = 0;
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
                    new_grid(i, j) = (Rand() >= exp(2.0 * K * E_site) ? grid(i,j) : -grid(i, j) );
                    //printf("Spin flipped! case 3. Probability = %.4f.\n", exp(2.0 * E_site)); //checkpoint
                }
                else
                {
                    new_grid(i, j) = grid(i, j);
                }
                //printf("Local energy = %.4e.\n", E_site); //checkpoint
            }
        }
        //Metropolis Algorithm to update the spin configuration on the new grid

        for (int i = halo; i < new_grid.xsize - halo; i++) //avoid double counting
        {
            for (int j = halo; j < new_grid.ysize - halo; j++)
            {
                E_new += -1.0 * new_grid(i, j) * (new_grid(i + 1, j) + new_grid(i, j + 1));
            }
        } //calculate the total energy of the new configuration

        if (round % (limit / 50) == 0) //report to screen every 100 round of evolution
        {
            fout << round << "\t" << E_new << endl;
            printf("Round %d finished. Current E = %.2f, last E = %.2f, difference = %.5f.\n", round, E_new, E_old, E_new - E_old);
        }
        grid = new_grid; //duplicate the current grid for updating
        round++;
    } while (std::abs(E_new - E_old) > epsilon && round < limit);

    fout.close();

    if (std::abs(E_new - E_old) <= epsilon)
    {
        printf("Energy converged, landscape mapped! total iteration: %d \n", round);
        new_grid.map("spin_map.dat", 0);
    }
    else if (round >= limit) //stop the program if it doesn't converge
    {
        printf("Evolution round exceeded the limit (%d rounds), simulation terminated and current spin configuration exported.\n", limit);
        new_grid.map("spin_map_current.dat", 0);
    }
    printf("Time used: %.2f seconds. \n", (float)((clock() - t_start) / CLOCKS_PER_SEC)); //print out total time lapsed

    return 0;
}