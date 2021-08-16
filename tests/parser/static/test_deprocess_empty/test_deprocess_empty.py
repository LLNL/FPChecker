import pathlib
import sys

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute())+"/../../../../parser")
#sys.path.insert(1, '/usr/workspace/wsa/laguna/fpchecker/FPChecker/parser')
from deprocess import Deprocess
from instrument import Instrument
from exceptions import EmptyFileException

FILE_PATH = str(pathlib.Path(__file__).parent.absolute())

SOURCE = FILE_PATH + "/MemUtils_HIP.cpp"
PRE_PROCESSED = FILE_PATH + "/MemUtils_HIP.cpp.o.ii"

def test_1():
  # This should produce an exception since the
  # de-preprocessed file is empty
  got_exception = False
  try:
    preFileName = PRE_PROCESSED
    sourceFileName = SOURCE
    inst = Instrument(preFileName, sourceFileName)
    inst.deprocess()
    inst.findDeviceDeclarations()
  except EmptyFileException:
    got_exception = True

  assert got_exception

if __name__ == '__main__':
  test_1()
