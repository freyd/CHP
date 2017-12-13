all: clean
	mpic++ grad_mpi2.cpp grad_conj.cpp -o grad
	mpiexec -np 4 --mca pml ob1 ./grad
clean:
	rm -f grad
