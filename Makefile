all: clean
	mpic++ grad_mpi.cpp grad_conj.cpp -o grad
	mpiexec -np 2 --mca pml ob1 ./grad
clean:
	rm -f grad
