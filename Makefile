# Pour compiler et exécuter le programme en parallèle :
# make all
# Pour compiler et exécuter le programme en séquentiel :
# make seq

all: clean
	mpic++ -std=c++11 -DPARALLEL grad_mpi2.cpp grad_conj.cpp -o grad
	mpiexec -np 5 --mca pml ob1 ./grad

seq: clean
	mpic++ -std=c++11 grad_mpi2.cpp grad_conj.cpp -o grad
	mpiexec -np 1 --mca pml ob1 ./grad
	
clean:
	rm -f grad
