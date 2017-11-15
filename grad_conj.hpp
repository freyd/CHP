#include <math.h>

void specialProd(int Nx, int Ny, double Lx, double Ly, double D, double dt, double* x, double *y);

void specialProd_parallel(int Nx, int Ny, double Lx, double Ly, double D, double dt, double* x, double *y,int rank,int size);

double prodscal(double *x, double *y, int taille);

void advance(int Nx, int Ny, double D, double Lx, double Ly, double dt, double *k, double *r, double *dd);

bool solve(int Nx,int Ny,int Nmax, double Lx, double Ly, double D, double eps, double dt, double *k, double *k0, double *b);
