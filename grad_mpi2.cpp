#include "grad_conj.hpp"
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// Fonction terme source
double func_f(double x, double y,double tmp){
  // return 2.0*(x-x*x + y-y*y);
  // return sin(x)+cos(y);
  return exp(-pow(x-0.5,2)-pow(y-0.5,2))*cos(3.141592654/2.*tmp);
}


// Fonction bord de gauche
double func_g(double x, double y){
  return 0.0;
  // return sin(x)+cos(y);
}


// Fonction bord de droite
double func_h(double x, double y){
  // return 0.0;
  // return sin(x)+cos(y);
  return 1.;
}


// Calcul de la norme
double norme_relative(double *k1, double * k2, int Nx, int Ny){
  int i,j;
  double norme = 0.0, norme2=0.0;
  for(i=0; i<Nx; i++)
    for(j=0; j<Ny; j++){
      norme += (k2[i*Ny+j]-k1[i*Ny+j])*(k2[i*Ny+j]-k1[i*Ny+j]);
      norme2 += k1[i*Ny+j]*k1[i*Ny+j];
    }
  return norme/norme2;
}



int main(int argc, char* argv[]){
  int Nx_global=120,Ny_global=100,Nx,Ny,Nmax=100;
  int i,j,itr,tps,max_iter=10,tag=99,recouv=2,p(0),q;
  double Lx = 1.0,Ly=1.0,D=1.0,eps=1e-6,dt=0.1,eps_schwarz=1e-6;
  double start,end;

  MPI::Init(argc, argv);

  MPI_Request request1, request2, request3, request4;
  MPI_Status status1, status2, status3, status4;

  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();

  start = MPI_Wtime();

  // Fonction charge
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

  printf("rank %d Nx %d\n",rank,Nx);

  // Prise en compte du recouvrement
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

  // Initialisation de la solution
  for(i=0;i<Nx;i++){
    for(j=0;j<Ny;j++){
      k[i*Ny + j] = 0.0;
      k_prec[i*Ny + j] = 0.0;
    }
  }

  // Initialisation du second membre b du système Ax = b
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

  // Affichage pourcentage
  if (rank == 0)
  {
    std::cout << p << " %" << std::endl;
  }
  double *k_schwarz = new double[Nx*Ny];

  int tmax=100;
  // Itérations en temps
  for(tps=0; tps<tmax; tps++){
    // Affichage pourcentage
    if (rank == 0)
    {
      q = floor((tps+1)*100./tmax);
      if (q != p)
      {
        p = q;
        std::cout << p << " %" << std::endl;
      }
    }

    // Itérations de Schwarz
    while(itr<500){

      // Sauvegarde de l'itération de Schwarz
      for(i=0;i<Nx;i++){
	       for(j=0;j<Ny;j++){
	          k_schwarz[i*Ny + j] = k[i*Ny + j];
	         }
         }

      // Résolution du gradient conjugué
      solve_parallel(Nx_global,Ny_global,Nmax,Lx,Ly,D,eps,dt,k,b,rank,size,recouv);
      if(rank==0)
	printf("norme relative: %lf\n",norme_relative(k_schwarz,k,Nx,Ny));

      // Critère d'arrêt de la boucle de Schwarz
      bool cond = (norme_relative(k_schwarz,k,Nx,Ny) < eps_schwarz);
      bool cond_global;
      MPI_Allreduce(&cond, &cond_global,1,MPI::BOOL,MPI_LAND,MPI_COMM_WORLD);
      if(cond_global){
	       if(rank==0)
	        printf("iteration finale schwarz %d\n",itr);
	         break;
      }

#ifdef PARALLEL
  // Communications aux frontières des sous-domaines
   if(rank != 0 && rank != size -1){
      MPI_Recv(buf1,Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status1);
      MPI_Send(&(k[(Nx-1-2*(recouv-1))*(Ny)]),Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD); // Modif Ny-1 ==> Ny Modif => 1* au lieu de 2*
      MPI_Recv(buf2,Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status2);
      MPI_Send(&(k[2*(recouv-1)*(Ny)]),Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
    }
    else{
      if(rank == 0){
	MPI_Send(&(k[(Nx-1-2*(recouv-1))*(Ny)]),Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
	MPI_Recv(buf2,Ny,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status1);

      }
      if(rank == size -1){
	MPI_Recv(buf1,Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status1);
	MPI_Send(&(k[2*(recouv-1)*(Ny)]),Ny,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
      }
    }

    // Mise à jour du second membre
    for(i=0;i<Nx;i++){
      for(j=0;j<Ny;j++){

	b[i*Ny +j] = dt*func_f( (rank*(Nx_global/size)+ std::min(rank,Nx_global%size) + (i+1))*dx,(j+1)*dy,tps*dt) + k_prec[i*Ny +j];

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
    }//Fin mise à jour de b
    itr++;
#endif

    }//Fin itérations de Schwarz

#ifdef PARALLEL
    // Sauvegarde de l'itération précédente
    for(i=0;i<Nx;i++){
      for(j=0;j<Ny;j++){
	k_prec[i*Ny +j] = k[i*Ny+j];
      }
    }
#endif

  }//Fin itérations en tps


  // Assemblage des solutions locales dans un vecteur global
  k_global = new double[Nx_global*Ny];

  int rcounts[size];
  int displs[size];
  for(int i=0; i<size;i++){
   if(i==0){
     displs[i] = 0;
       if(Nx_global%size == 0){
	 rcounts[i] = (Nx_global/size)*Ny;

       }
       else{
	 if(i < Nx_global%size){
	   rcounts[i] = (Nx_global/size + 1)*Ny;
	 }
	 else{
	   rcounts[i] = (Nx_global/size)*Ny;
	 }
       }
   }

   else{

       displs[i] = displs[i-1] + rcounts[i-1];

       if(Nx_global%size == 0){
	 rcounts[i] = (Nx_global/size)*Ny;

       }
       else{
	 if(i < Nx_global%size){
	   rcounts[i] = (Nx_global/size + 1)*Ny;
	 }
	 else{
	   rcounts[i] = (Nx_global/size)*Ny;
	 }

       }
   }
 }


  if(rank==0){
    MPI_Gatherv(k,(Nx-recouv+1)*Ny,MPI_DOUBLE,k_global,rcounts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
  }
  else
    if(rank != size-1)
      MPI_Gatherv(&(k[(recouv-1)*Ny]),(Nx- 2*(recouv-1))*Ny,MPI_DOUBLE,k_global,rcounts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    else
      MPI_Gatherv(&(k[(recouv-1)*Ny]),(Nx-(recouv-1))*Ny,MPI_DOUBLE,k_global,rcounts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);


    end = MPI_Wtime();

    // Fin algorithme

// Sauvegarde Paraview et Gnuplot
if(rank==0){

    // Sauvegarde Gnuplot
    FILE* output = fopen("graphe.txt","w+");
    err = 0.;
    for(i=0; i<Nx_global; i++){
      for(j=0; j<Ny; j++){
	fprintf(output,"%lf %lf %lf\n", k_global[i*Ny +j],(j+1)*dy,(i+1)*dx);

      // Calcul d'erreurs

	    //  err += pow((k_global[i*Ny +j] - (i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy)),2); // Cas 1
	     err += pow((k_global[i*Ny +j] - (sin((i+1)*dx) +cos((j+1)*dy))),2); // Cas 2
	    //  norm += pow((i+1)*dx*(1-(i+1)*dx)*(j+1)*dy*(1-(j+1)*dy),2);
	     norm += pow(sin((i+1)*dx) +cos((j+1)*dy),2);
      }
      }
      err /= norm;
      std::cout << err << "\n";
      fclose(output);

      // Sauvegarde Paraview (solution globale)
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




       // Sauvegarde Paraview (solution locale)
        std::ofstream solution;
          solution.open("graphe_"+to_string(rank)+".vtk", std::ios::out);
          solution << "# vtk DataFile Version 3.0" << endl;
          solution << "sol" << endl;
          solution << "ASCII" << endl;
          solution << "DATASET STRUCTURED_POINTS" << endl;
          solution << "DIMENSIONS " << Ny << " " << Nx << " " << 1 << endl;
          solution << "ORIGIN " << 0 << " " << 0 << " " << 0 << endl;
          solution << "SPACING " << dy << " " << dx << " " << 1 << endl;
          solution << "POINT_DATA " << Nx*Ny << endl;
          solution << "SCALARS sol float" << endl;
          solution << "LOOKUP_TABLE default" << endl;
          for(int i=0; i<Nx; ++i)
      {
        for(int j=0; j<Ny; ++j)
          {

            solution << k[i*Ny+j] << " ";
          }
        solution << endl;
      }
      solution.close();


// Affichage temps d'exécution
double delta,deltam;
delta = end-start;
MPI_Allreduce(&delta,&deltam,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

 if (rank == 0)
 {
   std::cout << "Temps d'exécution : " << delta << std::endl;
   std::cout << "Speed-up : " << 28.646/deltam << std::endl;
   std::cout << "Efficacité : " << 28.646/(size*deltam) << std::endl;

   FILE* output2 = fopen("eff_speedup.txt","a+");

 fprintf(output2,"%i %lf %lf %lf\n", size, deltam, 28.646/deltam, 28.646/(size*deltam));
     fclose(output2);
 }






  MPI::Finalize();
  delete[] k;
  delete[] b;
  delete[] k_prec;
  delete[] k_global;
  delete[] k_schwarz;
  return 0;
}
