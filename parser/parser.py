
import sys
from tokenizer import Tokenizer

if __name__ == '__main__':
  fileName = sys.argv[1]
  t = Tokenizer(fileName)
  for token in t.tokenize():
    print('token', type(token), ':', str(token), 'line:', token.lineNumber())
    #sys.stdout.write(str(token))

