#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>

enum class LatticeForm {
    square,
    kagome,
    triangular,
    circular,
    cubic,
    tetrahedral,
    spherical,
};

template <class T, LatticeForm format>
class lattice
{
    public: 
        std::vector<T> val;
        int dimension; //lattice dimension
        int xsize, ysize, zsize; //size of each dimension
        //string format; //lattice format: square, kagome, triangular

        lattice(); //Constructor
        lattice(int); 
        lattice(int, int); 
        lattice(int, int, int); //explicit constructor
       
        ~lattice(); //Destructor
       
        void set_value(int x, T value);
        void set_value(int x, int y, T value);
        void set_value(int x, int y, int z, T value);
        void map(std::string title, int offset);

        T& operator()(int x)
        {
            return val[x];
        } //calling for the value in the array
        
        T& operator()(int x, int y)
        {
            return val[index(x,y)];
        } //calling for the value in the array
        
        T& operator()(int x, int y, int z)
        {
            return val[index(x,y,z)];
        } //calling for the value in the array

        lattice<T, format>& operator=(const lattice<T, format> &input)
        {
            for (size_t i = 0; i < input.val.size(); ++i)
            {
                val[i] = input.val[i];
            }
            return *this;
        }

        template <class Type>
        inline auto index(Type x, Type y) //return vector index from input coordinates
        {
            switch (format)
            {
            case LatticeForm::square:
                return x + y * xsize;
            case LatticeForm::kagome:
                return x + y * xsize; 
            case LatticeForm::triangular:
                return x + y * xsize; 
            case LatticeForm::circular:
                return x + y * xsize; //to be defined later
            default:                  //default to square lattice
                return x + y * xsize;
            }
        }

        template <class Type>
        inline auto index(Type x, Type y, Type z) //return vector index from input coordinates
        {
            switch (format)
            {
            case LatticeForm::cubic:
                return x + y * xsize + z * xsize * ysize;
            case LatticeForm::spherical:
                return x + y * xsize; //to be defined later
            case LatticeForm::tetrahedral:
                return x + y * xsize + z * xsize * ysize;
            default:
                return x + y * xsize + z * xsize * ysize; //default to cubic lattice
            }
        }
};

template<class T, LatticeForm format>
lattice<T, format>::lattice():val(0) //default constructor
{
    std::printf("Default lattice of zero dimension initialized.\n");
}

template<class T, LatticeForm format>
lattice<T, format>::lattice(int x) : lattice(x, 1, 1)
{ }

template<class T, LatticeForm format>
lattice<T, format>::lattice(int x, int y) : lattice(x, y, 1)
{ }

template<class T, LatticeForm format>
lattice<T, format>::lattice(int x, int y, int z) :
    val(x * y * z, T()), xsize(std::move(x)),
    ysize(std::move(y)), zsize(std::move(z))
{
    if (x == 0 or y == 0 or z == 0)
    {
        throw std::runtime_error("For lower dimension, please define it explicitly!");
    } //avoid using "0" as argument to lower the dimension
    std::printf("Lattice of %d x %d x %d initialized.\n", xsize, ysize, zsize);
}

template<class T, LatticeForm format>
lattice<T, format>::~lattice() = default;

template<class T, LatticeForm format>
void lattice<T, format>::set_value(int x, T value)
{
    val[x] = value;
}

template<class T, LatticeForm format>
void lattice<T, format>::set_value(int x, int y, T value)
{
    val[index(x,y)] = value;
}

template<class T, LatticeForm format>
void lattice<T, format>::set_value(int x, int y, int z, T value)
{
    val[index(x,y,z)] = value;
}

template<class T, LatticeForm format>
void lattice<T, format>::map(const std::string title, const int offset)
{
		std::string newname = title + ".dat";
		std::ofstream fout;
		fout.precision(6);
		fout.open(newname.c_str());
		for (int j = offset; j < (ysize - offset); ++j)
		{
			for (int i = offset; i < (xsize - offset); ++i)
			{
				fout << val[index(i, j)] << "\t";
			}
			fout << std::endl;
		}
        fout.close();
}