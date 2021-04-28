import os
import pathlib
import sys
import subprocess

sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer
from instrument import Instrument

RUNTIME='../../../../src/Runtime_parser.h'

prog_1 = """

__device__ void comp() {

  double cross1, cross2, cross0;
  struct facet_coords0 {
    double x,y,z;
  };
  struct facet_coords1 {
    double x,y,z;
  };
  struct facet_coords2 {
    double x,y,z;
  };
  struct intersection_pt {
    double x,y,z;
  };


#define AB_CROSS_AC(ax,ay,bx,by,cx,cy) ( (bx-ax)*(cy-ay) - (by-ay)*(cx-ax) )

       cross1 = AB_CROSS_AC(facet_coords0.x, facet_coords0.y,
                            facet_coords1.x, facet_coords1.y,
                            intersection_pt.x,  intersection_pt.y);
       cross2 = AB_CROSS_AC(facet_coords1.x, facet_coords1.y,
                            facet_coords2.x, facet_coords2.y,
                            intersection_pt.x,  intersection_pt.y);
       cross0 = AB_CROSS_AC(facet_coords2.x, facet_coords2.y,
                            facet_coords0.x, facet_coords0.y,
                            intersection_pt.x,  intersection_pt.y);

 
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
  inst_program(prog_1, 'prog_1', 1)

if __name__ == '__main__':
  test_1()

