#include "grad_conj.hpp"
#include "mpi.h"
#include <iostream>

int main(int argc, char* argv[]){

  MPI::Init(argc, argv);
  
  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();
  std::cout << "I am " << rank << std::endl;
  
  
  MPI::Finalize();
  return 0;
}
