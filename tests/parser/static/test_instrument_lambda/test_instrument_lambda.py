import os
import pathlib
import sys
import subprocess

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer

source = 'compute_inst.cu'

def setup_module(module):
  THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  os.chdir(THIS_DIR)
 
def teardown_module(module):
  cmd = ["make clean"]
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
  cmd = ["make"]
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

  # it should find one instrumented statement 
  numbwerOfTransformations = 0
  fd = open(source, "r")
  for l in fd:
    if "_FPC_CHECK_HD_(" in l:
      numbwerOfTransformations = numbwerOfTransformations + 1
  fd.close()
     
  assert numbwerOfTransformations == 1

if __name__ == '__main__':
  test_1()

