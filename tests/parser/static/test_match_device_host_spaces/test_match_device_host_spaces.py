
import sys
import pathlib
import subprocess

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from tokenizer import KeywordToken, SymbolToken, WhiteSpaceToken, IdentifierToken
from match import Match

PATH = str(pathlib.Path(__file__).parent.absolute())

def teardown_module(module):
  print("In teardown_module...")
  try:
    #cmd = ["cd "+PATH]
    #cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    cmd = ["cd " + PATH + " && rm -f *.o *.ii"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    print(e.output)
    exit()

def test_1():
 
  t1 = KeywordToken('__attribute__', 1)
  t2 = SymbolToken('(', 1)
  t3 = SymbolToken('(', 1)
  t4 = IdentifierToken('device', 1)
  t5 = SymbolToken(')', 1)
  t6 = SymbolToken(')', 1)
  t7 = WhiteSpaceToken(' ', 1)

  buff = []
  buff.append(t1)
  buff.append(t2)
  buff.append(t3)
  buff.append(t4)
  buff.append(t5)
  buff.append(t6)
  buff.append(t7)

  m = Match()
  d = m._match_device_decl(buff)
  print(d)
  assert d == 6

  # device host
  buff.append(WhiteSpaceToken(' ', 1))
  buff.append(WhiteSpaceToken(' ', 1))
  buff.append(KeywordToken('__attribute__', 1))
  buff.append(SymbolToken('(', 1))
  buff.append(SymbolToken('(', 1))
  buff.append(IdentifierToken('host', 1))
  buff.append(SymbolToken(')', 1))
  buff.append(SymbolToken(')', 1))

  d_h = m._match_device_host_decl(buff)
  print(d_h)
  assert d_h == 15

  # host device
  buff[3] = IdentifierToken('host', 1)
  buff[12] = IdentifierToken('device', 1)
  
  h_d = m._match_host_device_decl(buff)
  print(h_d)
  assert d_h == 15
