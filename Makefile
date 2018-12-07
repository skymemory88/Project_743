OPTIM+=-O3 -march=native
CXX=g++ #mpicxx/mpiicpc
CC=g++
CXXFLAGS+= -Wall -Wextra -std=c++14 $(OPTIM) -g #-fopenmp
EXE=Ising_model_serial_boolean

all: clean $(EXE)

Ising_model: Ising_model_serial_boolean.o mtrand.hpp Lattice.hpp

clean:
	rm -f $(EXE) Ising_model_serial_boolean.o 2>&-
