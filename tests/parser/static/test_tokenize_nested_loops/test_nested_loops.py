import subprocess
import os
import pathlib
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer

PROGRAM =   """
__device__ void calc(double *x, int N) {
 for(int i=0; i<N; ++i) {
  for (int j=0; j<3; ++j) {
   x[i][j] = 0.0;
  }
 }
}
"""

SOURCE = "tmp.cu"

def setup_module(module):
  THIS_DIR = os.path.dirname(os.path.abspath(__file__))
  os.chdir(THIS_DIR)
  with open(SOURCE, 'w') as fd:
    for l in PROGRAM:
      fd.write(l)
 
def teardown_module(module):
  cmd = ["rm -f "+SOURCE]
  cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
  l = Tokenizer(SOURCE)
  count = 0
  for token in l.tokenize():
    count += 1
    sys.stdout.write('\n'+str(type(token))+':')
    sys.stdout.write(str(token))
  print('Len:', count)

  assert count == 90

if __name__ == '__main__':
  test_1()
