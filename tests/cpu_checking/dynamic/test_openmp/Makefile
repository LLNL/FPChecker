
CLANG_PATH = $(shell which clang)
OMP_LIB = $(shell dirname $(CLANG_PATH))/../lib
 
OP = 	-O2 -fopenmp
CXX=FPC_VERBOSE=1 FPC_INSTRUMENT=1 clang++-fpchecker 

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute.cpp $(OP)
	$(CXX) -L$(OMP_LIB) -o main compute.o main.o -lomp

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
