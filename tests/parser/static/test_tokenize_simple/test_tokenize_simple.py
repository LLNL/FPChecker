import pathlib
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer

FILENAME = str(pathlib.Path(__file__).parent.absolute()) + "/simple.cu"

def test_1():
  l = Tokenizer(FILENAME)
  count = 0
  for token in l.tokenize():
    count += 1
    sys.stdout.write('\n'+str(type(token))+':')
    sys.stdout.write(str(token))
  print('Len:', count)

  assert count == 28

