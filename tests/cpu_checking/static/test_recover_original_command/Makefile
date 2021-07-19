
OP = 	-O0 
CXX = FPC_VERBOSE=1 FPC_INJECT_FAULT=1 FPC_INSTRUMENT=1 clang++-fpchecker

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -o main main.o

clean:
	rm -rf *.o main __pycache__ .fpc_traces *.o.tmp .fpc_log.txt
