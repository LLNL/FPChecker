include ../../Makefile.config

OP	= $(LLVM_PASS) -O0 -g -fvisibility=hidden

all:
	$(CC) -c main.cpp $(OP)
	$(CC) -o main main.o

clean:
	rm -rf *.o main __pycache__
