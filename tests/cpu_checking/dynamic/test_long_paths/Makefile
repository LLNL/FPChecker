
OP = 	-O2 
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 

FILE=long_path/long_path/long_path/long_path/long_path/long_path/long_path/long_path/long_path/long_path/long_path/long_path/compute.cpp

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c $(FILE) $(OP)
	$(CXX) -o main compute.o main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
