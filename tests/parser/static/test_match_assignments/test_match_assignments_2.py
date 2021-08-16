import os
import pathlib
import sys
import subprocess

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer
from instrument import Instrument

RUNTIME='../../../../src/Runtime_parser.h'

prog_2 = """  
__device__ double *p_new;
__device__ double *bvc;
__device__ double *e_old;
__device__ double fabs(double x);
__device__ double p_cut;
__device__ double *vnewc;
__device__ double eosvmax;
__device__ double pmin;

 __device__ void comp(int i) {
  p_new[i] = bvc[i] * e_old[i] ; if ( fabs(p_new[i]) < p_cut ) p_new[i] = 0.0 ; if ( vnewc[i] >= eosvmax ) p_new[i] = 0.0 ; if ( p_new[i] < pmin ) p_new[i] = pmin ;;
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
    assert n == num_inst
    return True
  except Exception as e:
    print(e)
    return False

def test_1():
  os.environ['FPC_VERBOSE'] = '1'
  assert inst_program(prog_2, 'prog_2', 1)

if __name__ == '__main__':
  test_1()

