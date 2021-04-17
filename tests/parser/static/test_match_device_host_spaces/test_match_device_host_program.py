import pathlib
import sys
import subprocess

sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer
from nvcc_fpchecker import Command

PATH = str(pathlib.Path(__file__).parent.absolute())

def test_1():
  passed = True
  try:
    strCmd = "nvcc --std=c++11 -arch=sm_70 -O0 -c " + PATH+"/dot_product.cu " + "-o " + PATH+"/dot_product.o"
    cmd = Command(strCmd.split())
    cmd.executePreprocessor()
    cmd.instrumentSource()
    cmd.compileInstrumentedFile()
  except Exception as e:
    passed = False

  assert passed

if __name__ == '__main__':
  test_1()
