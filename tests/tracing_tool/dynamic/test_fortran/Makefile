
all:
	cc -c main.c -O1 -I./
	gfortran -c compute.f95 
	gfortran -o main main.o compute.o

clean:
	rm -rf *.o main __pycache__ *_copy.* main_fpc
