OPTIM+=-O3 -march=native
CXX=g++ #mpicxx/mpiicpc
CC=g++
CXXFLAGS+= -Wall -Wextra -fopenmp -std=c++14 $(OPTIM) -g
EXE=Ising_model_serial

all: clean $(EXE)

Ising_model: Ising_model_serial.o mtrand.hpp Lattice.hpp

clean:
	rm -f $(EXE) Ising_model_serial.o 2>&-
