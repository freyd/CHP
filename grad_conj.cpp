#include "grad_conj.hpp"
#include <stdio.h>

void specialProd(int Nx, int Ny, double Lx, double Ly, double D, double dt, double* x, double *y){
  int i,j,k;
  double dx,dy,alpha,beta,gamma;
  dx = Lx/(1.0 + Nx);
  dy = Ly/(1.0 + Ny);

  alpha = 1.0 + 2.0*dt*D*(1.0/(dx*dx) + 1.0/(dy*dy));
  beta = - dt*D*1.0/(dx*dx);
  gamma = - dt*D*1.0/(dy*dy);
  printf("alpha : %lf gamma : %lf beta : %lf\n", alpha,gamma,beta);
  for(j=0; j<Ny;j++)
    for(i=0; i<Nx; i++){
      k = j + Ny*(i);
      y[k] = alpha*x[k];
      if (j != (Ny-1)) 
	y[k] = y[k] + gamma*x[k+1];
      if (j != 0) 
	y[k] = y[k] + gamma*x[k-1];
      if (i != (Nx-1)) 
	y[k] = y[k] + beta*x[k+Ny];
      if (i != 0) 
	y[k] = y[k] + beta*x[k-Ny];
    }

};

void specialProd_parallel(int Nx, int Ny, double Lx, double Ly, double D, double dt, double* x, double *y, int rank, int size){
  int i,j,k;
  int debut, fin;
  double dx,dy,alpha,beta,gamma;
  dx = Lx/(1.0 + Nx);
  dy = Ly/(1.0 + Ny);

  alpha = 1.0 + 2.0*dt*D*(1.0/(dx*dx) + 1.0/(dy*dy));
  beta = - dt*D*1.0/(dx*dx);
  gamma = - dt*D*1.0/(dy*dy);
  // printf("alpha : %lf gamma : %lf beta : %lf\n", alpha,gamma,beta);

  if(Ny%size == 0){
    debut = rank*(Ny/size);
    fin = (rank+1)*(Ny/size);
  }
  else{
    if(rank < Ny%size){
      debut = rank*(Ny/size +1);
      fin = (rank+1)*(Ny/size +1);
    }
    else{
      debut = (Ny%size)*(Ny/size + 1) + (rank-(Ny%size))*(Ny/size);
      fin = (Ny%size)*(Ny/size + 1) + (rank-(Ny%size) + 1)*(Ny/size);
    }
  }

  
  for(i=0; i< Nx; i++)
    for(j= debut; j<fin;j++){
      k = j + Ny*(i);
      y[k] = alpha*x[k];
      if (j != (Ny-1)) 
	y[k] = y[k] + gamma*x[k+1];
      if (j != 0) 
	y[k] = y[k] + gamma*x[k-1];
      if (i != (Nx-1)) 
	y[k] = y[k] + beta*x[k+Ny];
      if (i != 0) 
	y[k] = y[k] + beta*x[k-Ny];
    }

};

double prodscal(double *x, double *y, int taille){
  double r = 0;
  int i;
  for(i=0;i<taille;i++)
    r += x[i]*y[i];
  return r;
};


void advance(int Nx, int Ny, double D, double Lx, double Ly, double dt, double *k, double *r, double *dd){
  double alpha, beta;
  double *w = new double[Nx*Ny];
  double *p = new double[Nx*Ny];
  int i;

  specialProd(Nx,Ny,Lx,Ly,D,dt,dd,w);
  alpha = prodscal(dd,r,Nx*Ny)/prodscal(dd,w,Nx*Ny);
  
  for(i=0;i<Nx*Ny;i++){
    k[i] = k[i] - alpha*dd[i];
    p[i] = r[i] - alpha*w[i];
  }
  beta = prodscal(p,p,Nx*Ny)/prodscal(r,r,Nx*Ny);
  
  for(i=0;i<Nx*Ny;i++){
    r[i] = p[i];
    dd[i] = r[i] + beta*dd[i];
  }
  delete[] p;
  delete[] w;
}; 

bool solve(int Nx,int Ny,int Nmax, double Lx, double Ly, double D, double eps, double dt, double *k, double *k0, double *b){
  int i;
  double *r = new double[Nx*Ny];
  double *dd = new double[Nx*Ny];
  double norme;

  for(i=0;i<Nx*Ny;i++)
    k[i] = k0[i];
  
  specialProd(Nx,Ny,Lx,Ly,D,dt,k,r);
  
  for(i=0;i<Nx*Ny;i++){
    r[i] = r[i] - b[i];
    dd[i] = r[i];
  }

  i = 0;
  norme = sqrt(prodscal(r,r,Nx*Ny));

  while(i<=Nmax && norme>eps)
    {
      advance(Nx,Ny,D,Lx,Ly,dt,k,r,dd);
      i++;
      norme = sqrt(prodscal(r,r,Nx*Ny));
    }
  
  return (norme < eps);
  delete[] r;
  delete[] dd;
};
/*
int main(){
  int Nx = 2;
  int Ny = 2;
  int Nmax = 2;
  double Lx = 1.0;
  double Ly = 1.0;
  double D = 1.0;
  double eps = 1e-5;
  double dt = 1.0;
  double *k = new double[Nx*Ny];
  double *k0 = new double[Nx*Ny];
  double *b = new double[Nx*Ny];
  int i;

  for(i=0;i<Nx*Ny;i++){
    k[i] = 1.0;
    k0[i] = 1.0;
    b[i] = 1.0;
  }
  
  solve(Nx,Ny,Nmax,Lx,Ly,D,eps,dt,k,k0,b);
  for(i=0;i<Nx*Ny;i++)
    printf("%lf ", k[i]);  
  printf("\n");
}
*/
