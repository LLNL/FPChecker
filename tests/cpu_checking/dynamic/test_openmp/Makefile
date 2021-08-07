
OP = 	-O2 -fopenmp
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute.cpp $(OP)
	$(CXX) -o main compute.o main.o -fopenmp

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
