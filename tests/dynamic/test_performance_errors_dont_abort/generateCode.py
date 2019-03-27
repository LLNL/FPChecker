
import sys

def genCode(lines):
    fd = open('arr_sum.cu', 'w')
    fd.write("#include <stdio.h>\n")
    fd.write("__global__ void array_sum(double *x, double *y) {\n")
    fd.write("int i=0;\n")
    for i in range(lines):
        if i % 2 == 0:
            fd.write("\tx[i] += y[i];\n")
        else:
            fd.write("\tx[i] -= y[i];\n")
    fd.write("}\n")
    fd.close()

if __name__ == "__main__":
    genCode(int(sys.argv[1]))
