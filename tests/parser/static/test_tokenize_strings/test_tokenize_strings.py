
import sys

sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import Tokenizer

FILENAME = "simple.cu"

def test_1():
  l = Tokenizer(FILENAME)
  count = 0
  for token in l.tokenize():
    count += 1
    sys.stdout.write('\n'+str(type(token))+':')
    sys.stdout.write(str(token))
  print('Len:', count)


if __name__ == '__main__':
  test_1()
