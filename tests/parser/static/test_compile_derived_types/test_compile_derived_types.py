
import sys
import subprocess

sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer

FILENAME = "simple.cu"

def test_1():

  try:
    cmdOutput = subprocess.run('make clean && make', shell=True, check=True)
  except Exception as e:
    print(e)
 
if __name__ == '__main__':
  test_1()
