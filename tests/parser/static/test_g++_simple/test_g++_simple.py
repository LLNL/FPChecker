import os
import pathlib
import sys
import subprocess

sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer
from instrument import Instrument

RUNTIME='../../../../src/Runtime_g++.h'

prog_1 = """
double foo(double a);

void comp(double x) {
  if (x = foo(3.0))
    x = x*x;
}
"""

def setup_module(module):
  THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  os.chdir(THIS_DIR)
 
def teardown_module(module):
  cmd = ["rm -f *.o *.ii *.cpp"]
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def preprocessFile(prog_name: str):
  cmd = ['g++ -E '+prog_name+'.cpp -o '+prog_name+'.ii']
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def createFile(prog: str, prog_name: str):
  with open(prog_name+'.cpp', 'w') as fd:
    fd.write(prog)
    fd.write('\n')
  preprocessFile(prog_name)

def instrument(prog_name: str):
  pass
  preFileName = prog_name+'.ii'
  sourceFileName = prog_name+'.cpp'
  inst = Instrument(preFileName, sourceFileName)
  inst.deprocess()
  inst.findAllDeclarations()
  inst.findAssigments()
  inst.produceInstrumentedLines()
  inst.instrument()

def compileProggram(prog_name: str):
  cmd = ['g++ -std=c++11 -c -include '+RUNTIME+' '+prog_name+'_inst.cpp']
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def countInstrumentationCalls(prog_name: str):
  ret = 0
  with open(prog_name+'_inst.cpp', 'r') as fd:
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
  assert inst_program(prog_1, 'prog_1', 2)

if __name__ == '__main__':
  test_1()

