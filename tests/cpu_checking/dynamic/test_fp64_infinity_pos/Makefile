
OP = 	-O2
CXX = FPC_INSTRUMENT=1 ~/Documents/Tools/FPChecker/build/bin/clang++-fpchecker -DFPC_POSITIVE_OVERFLOW=1

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute.cpp $(OP)
	$(CXX) -o main compute.o main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
