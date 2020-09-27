
all:
	xlc -c main.c -O1 -I./
	xlf -c compute.f95 
	xlc -o main main.o compute.o

clean:
	rm -rf *.o main __pycache__ *_copy.* main_fpc
