import os
import pathlib
import sys
import subprocess

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer
from instrument import Instrument

RUNTIME='../../../../src/Runtime_parser.h'

prog_1 = """
__device__ void comp() {

  double dx31, dx72, dx63, dx20, dx43, dx57, dx64, dx70, dx14, dx25, dx61, dx50, dz43, dz57, dz72, dz64, dz50, dz61;
  double dy31, dy72, dy63, dy20, dy43, dy57, dy64, dy70, dy14, dy25, dy61, dy50, dz20, dz31, dz70, dz25, dz63, dz14;
  double twelveth;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  double volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;
}
"""

prog_2 = """
#define MY_MACRO(x1, x2, x3, x4, x5) (x1+x2+x3+x4)

__device__ void compute(double *y, double a1, double a2, double a3, double a4, double a5) {
  y[0] = MY_MACRO(a1,
        a2,
        a3,
        a4,
        a5);
}
"""

def setup_module(module):
  THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  os.chdir(THIS_DIR)
 
def teardown_module(module):
  cmd = ["rm -f *.o *.ii *.cu"]
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def preprocessFile(prog_name: str):
  cmd = ['nvcc -E '+prog_name+'.cu -o '+prog_name+'.ii']
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def createFile(prog: str, prog_name: str):
  with open(prog_name+'.cu', 'w') as fd:
    fd.write(prog)
    fd.write('\n')
  preprocessFile(prog_name)

def instrument(prog_name: str):
  pass
  preFileName = prog_name+'.ii'
  sourceFileName = prog_name+'.cu'
  inst = Instrument(preFileName, sourceFileName)
  inst.deprocess()
  inst.findDeviceDeclarations()
  inst.findAssigments()
  inst.produceInstrumentedLines()
  inst.instrument()

def compileProggram(prog_name: str):
  cmd = ['nvcc -std=c++11 -c -include '+RUNTIME+' '+prog_name+'_inst.cu']
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def countInstrumentationCalls(prog_name: str):
  ret = 0
  with open(prog_name+'_inst.cu', 'r') as fd:
    for l in fd.readlines():
      for w in l.split():
        if '_FPC_CHECK_' in w:
          ret += 1
  return ret

def inst_program(prog: str, prog_name: str, num_inst: int):
  try:
    createFile(prog, prog_name)
    instrument(prog_name)
    compileProggram(prog_name)
    n = countInstrumentationCalls(prog_name)
    #assert n == num_inst
    return True
  except Exception as e:
    print(e)
    return False

def test_1():
  os.environ['FPC_VERBOSE'] = '1'
  assert inst_program(prog_1, 'prog_1', 2)
  assert inst_program(prog_2, 'prog_2', 1)

if __name__ == '__main__':
  test_1()

