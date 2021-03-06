#include "grad_conj.hpp"
#include "mpi.h"
#include <iostream>

double func_f(double x, double y){
  return 2.0*(x-x*x + y-y*y);
}

double func_g(double x, double y){
  return 0.0;
}

double func_h(double x, double y){
  return 0.0;
}

int main(int argc, char* argv[]){
  int Nx_global=4,Ny_global=5,Nx,Ny,Nmax=100;
  int i,j,itr,tps,max_iter=10,tag=99;
  double Lx = 1.0,Ly=1.0,D=1.0,eps=1e-6,dt=0.1;
  double start, end;
  MPI::Init(argc, argv);
  MPI_Request request1, request2, request3, request4;
  MPI_Status status1, status2, status3, status4;

  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();

  start = MPI_Wtime();

  Ny = Ny_global;
  if(Nx_global%size == 0){
    Nx = Nx_global/size;
  }
  else{
    if(rank < Nx_global%size){
      Nx = Nx_global/size +1;
    }
    else{
      Nx = Nx_global/size;
    }
  }

  /* if(rank!=1)
    Nx++;
  */
  double *k = new double[Nx*Ny];
  double *k_prec = new double[Nx*Ny];

double *b = new double[Nx*Ny];
  double dx = Lx/(1.0 + Nx_global);
  double dy = Ly/(1.0 + Ny_global);
  double beta = - dt*D*1.0/(dx*dx);
  double gamma = - dt*D*1.0/(dy*dy);
  double *buf1 = new double[Ny];
  double *buf2 = new double[Ny];

  for(i=0;i<Nx;i++){
    for(j=0;j<Ny;j++){
      k[i*Ny + j] = 0.0;
      k_prec[i*Ny + j] = 0.0;
    }
  }


  for(i=0;i<Nx;i++){
    for(j=0;j<Ny;j++){

      b[i*Ny +j] = dt*func_f( (rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,(j+1)*dy);

      if(j==0)
	b[i*Ny +j] -= gamma*func_g((rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,0.0);

      if(j==Ny-1)
	b[i*Ny +j] -= gamma*func_g((rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,Ly);

      if(i==0){
	if(rank==0)
	  b[i*Ny +j] -= beta*func_h(0.0,(j+1)*dy);
      }

      if(i==Nx-1){
	if(rank==size-1)
	  b[i*Ny +j] -= beta*func_h(Lx,(j+1)*dy);
      }
    }
  }

  if(rank==0){
  for(i=0; i<Nx; i++){
    for(j=0; j<Ny; j++)
      std::cout << b[i*Ny +j] << " ";
    std::cout << "\n";
  }
  }

  for(tps=0; tps<10; tps++){
  //itérations de Schwartz
    for(itr=0;itr<200;itr++){

      solve_parallel(Nx_global,Ny_global,Nmax,Lx,Ly,D,eps,dt,k,b,rank,size);

 if(rank != 0 && rank != size -1){
      MPI_Recv(buf1,Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status1);
      MPI_Send(&(k[(Nx-2-2*(recouv-1))*(Ny-1)]),Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
      MPI_Recv(buf2,Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status2);
      MPI_Send(&(k[2*(recouv-1)*(Ny-1)]),Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
    }
    else{
      if(rank == 0){
	MPI_Send(&(k[(Nx-2-2*(recouv-1))*(Ny-1)]),Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
	MPI_Recv(buf2,Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status1);

      }
      if(rank == size -1){
	MPI_Recv(buf1,Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status1);
	MPI_Send(&(k[2*(recouv-1)*(Ny-1)]),Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
      }
    }



    for(i=0;i<Nx;i++){
      for(j=0;j<Ny;j++){

      b[i*Ny +j] = dt*func_f( (rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,(j+1)*dy) + k_prec[i*Ny +j];

      if(j==0)
	b[i*Ny +j] -= gamma*func_g((rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,0.0);

      if(j==Ny-1)
	b[i*Ny +j] -= gamma*func_g((rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,Ly);

      if(i==0){
	if(rank==0)
	  b[i*Ny +j] -= beta*func_h(0.0,(j+1)*dy);
	else
	  b[i*Ny +j] -= beta*buf1[j];
      }

      if(i==Nx-1){
	if(rank==size-1)
	  b[i*Ny +j] -= beta*func_h(Lx,(j+1)*dy);
	else
	  b[i*Ny +j] -= beta*buf2[j];
      }
      }
    }//modif b

    }//iter schwarz

    for(i=0;i<Nx;i++){
      for(j=0;j<Ny;j++){
	k_prec[i*Ny +j] = k[i*Ny+j];
      }
    }


  }//iter tps

  end = MPI_Wtime();
  //std::cout << "I am " << rank << std::endl;
    std::cout << "\n";

       if(rank==0){
    for(i=0; i<Nx; i++){
      for(j=0; j<Ny; j++)
	std::cout << (k[i*Ny +j] - (i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy))/(i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy) << " ";
      std::cout << "\n";
    }
    }

  std::cout << "Temps d'exécution : " << end-start << std::endl;


  MPI::Finalize();
  delete[] k;
  delete[] b;
  delete[] k_prec;
  return 0;
}
