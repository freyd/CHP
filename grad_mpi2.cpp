#include "grad_conj.hpp"
#include "mpi.h"
#include <iostream>
#include <fstream>
using namespace std;
double func_f(double x, double y,double tmp){
  return 2.0*(x-x*x + y-y*y);
}

double func_g(double x, double y){
  return 0.0;
}

double func_h(double x, double y){
  return 0.0;
}

int main(int argc, char* argv[]){
  int Nx_global=20,Ny_global=25,Nx,Ny,Nmax=100;
  int i,j,itr,tps,max_iter=10,tag=99,recouv=2;
  double Lx = 1.0,Ly=1.0,D=1.0,eps=1e-6,dt=0.1;
  MPI::Init(argc, argv);
  MPI_Request request1, request2, request3, request4;
  MPI_Status status1, status2, status3, status4;

  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();
  
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

   if(rank!=0 && rank != size-1)
     Nx+= 2*(recouv-1);
   else
     Nx+= recouv-1;
  
  double *k = new double[Nx*Ny];
  double *k_prec = new double[Nx*Ny];

double *b = new double[Nx*Ny];
  double dx = Lx/(1.0 + Nx_global);
  double dy = Ly/(1.0 + Ny_global);
  double beta = - dt*D*1.0/(dx*dx);
  double gamma = - dt*D*1.0/(dy*dy);
  double *buf1 = new double[Ny]; 
  double *buf2 = new double[Ny]; 
  double * k_global;

  for(i=0;i<Nx;i++){
    for(j=0;j<Ny;j++){
      k[i*Ny + j] = 0.0;
      k_prec[i*Ny + j] = 0.0;
    }
  }


  for(i=0;i<Nx;i++){
    for(j=0;j<Ny;j++){

      b[i*Ny +j] = dt*func_f( (rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,(j+1)*dy, 0.0);

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


  for(tps=0; tps<10; tps++){
  //itérations de Schwartz
    for(itr=0;itr<300;itr++){
      
      solve_parallel(Nx_global,Ny_global,Nmax,Lx,Ly,D,eps,dt,k,b,rank,size,recouv);
    #ifdef PARALLEL
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

	b[i*Ny +j] = dt*func_f( (rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,(j+1)*dy,tps) + k_prec[i*Ny +j];

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
#endif
    }//iter schwarz
#ifdef PARALLEL
    for(i=0;i<Nx;i++){
      for(j=0;j<Ny;j++){
	k_prec[i*Ny +j] = k[i*Ny+j];
      }
    }
#endif

  }//iter tps
  //std::cout << "I am " << rank << std::endl;

 /* 
 int rcounts[size];
 int displs[size];
 for(int i=0; i<size;i++){
   if(i==0){
     displs[i] = 0;
     rcounts[i] = (Nx_global/size + recouv-1)*Ny;
   }
   else{
     if(i == size-1){
       displs[i] = displs[i-1] + (Nx_global/size + 2*(recouv-1))*Ny;
       rcounts[i] = (Nx_global/size + recouv-1)*Ny;
     }
     else{
       if(i==1){
	 displs[i] = displs[i-1] + (Nx_global/size + (recouv-1))*Ny;
       }
       else
	 displs[i] = displs[i-1] + (Nx_global/size + 2*(recouv-1))*Ny;
       rcounts[i] = (Nx_global/size + 2*(recouv-1))*Ny;
     }
   }
 }
 
 MPI_Gatherv(k,Nx*Ny,MPI_DOUBLE,k_global,rcounts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);*/


  if(rank==0){ 
    k_global = new double[Nx_global*Ny];
    MPI_Gather(k,(Nx_global/size)*Ny,MPI_DOUBLE,k_global,(Nx_global/size)*Ny,MPI_DOUBLE,0,MPI_COMM_WORLD);
  } 
  else
    MPI_Gather(&(k[recouv-1]),(Nx_global/size)*Ny,MPI_DOUBLE,k_global,(Nx_global/size)*Ny,MPI_DOUBLE,0,MPI_COMM_WORLD);

   
if(rank==0){
  //std::cout << "\n";
    FILE* output = fopen("graphe.txt","w+");

    for(i=0; i<Nx_global; i++){
      for(j=0; j<Ny; j++){
	fprintf(output,"%lf %lf %lf\n", k_global[i*Ny +j],(j+1)*dy,(i+1)*dx);
	//std::cout << (k_global[i*Ny +j] - (i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy))/(i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy) << " ";
      }    
      //std::cout << "\n"; 
      }
      fclose(output);
     std::ofstream solution;
       solution.open("graphe.vtk", std::ios::out);
       solution << "# vtk DataFile Version 3.0" << endl;
       solution << "sol" << endl;
       solution << "ASCII" << endl;
       solution << "DATASET STRUCTURED_POINTS" << endl;
       solution << "DIMENSIONS " << Ny << " " << Nx_global << " " << 1 << endl;
       solution << "ORIGIN " << 0 << " " << 0 << " " << 0 << endl;
       solution << "SPACING " << dy << " " << dx << " " << 1 << endl;;
       solution << "POINT_DATA " << Nx_global*Ny << endl;
       solution << "SCALARS sol float" << endl;
       solution << "LOOKUP_TABLE default" << endl;
       for(int i=0; i<Nx_global; ++i)
	 {
	   for(int j=0; j<Ny; ++j)
	     {
	       
	       solution << k_global[i*Ny+j] << " ";
	     }
	   solution << endl;
	 }
	 solution.close();
       }


 

  MPI::Finalize();
  delete[] k;
  delete[] b;
  delete[] k_prec;
  if(rank==0)
    delete[] k_global;
  return 0;
}
