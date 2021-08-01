#!/usr/bin/env python3

import subprocess
import sys
import os
from colors import *

#INTERCEPT_LIB = '/usr/workspace/wsa/laguna/fpchecker/FPChecker/interception_tool/intercept.so'
INTERCEPT_LIB = os.path.dirname(os.path.abspath(__file__))+"/../lib/libfpchecker_intercept_lib.so"

def runBuildCommand(params):
  prGreen('*** FPChecker ***')
  prGreen('Intercepting commands in: ' + ' '.join(params))
  params.insert(0, 'FPC_INSTRUMENT=1')
  params.insert(0,'LD_PRELOAD='+INTERCEPT_LIB)  

  try:
    cmdOutput = subprocess.run(' '.join(params), shell=True, check=True)
  except Exception as e:
    print(e)
    raise RuntimeError('Error when running fpchecker input')

if __name__ == '__main__':
  params = sys.argv
  params.pop(0)
  runBuildCommand(params)
