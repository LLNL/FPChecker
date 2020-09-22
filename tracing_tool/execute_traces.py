import sys
import subprocess
from colors import prRed 

def execTraces(fileName):
  fd = open(fileName, 'r')
  for cmd in fd:
    try:
      print( cmd)
      cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
      print(cmdOutput.decode('utf-8'))
    except subprocess.CalledProcessError as e:
      prRed('Error:')
      print(e.output.decode('utf-8')) 
      exit(-1)
  fd.close()

if __name__ == '__main__':
  fileName = sys.argv[1]
  print('Executing', fileName)
  execTraces(fileName)
