import subprocess
import sys
import os

class MPIEnvironment:

  def __init__(self, nvcc_command):
    self.includeDir = []
    options = nvcc_command.split()

    # Find -ccbin options
    self.compiler = None
    for i in range(len(options)):
      op = options[i]
      if '-ccbin' in op or '--compiler-bindir' in op:
        if '=' in op: # separated by =
          self.compiler = op.split('=')[1]
        else:         # separated by space
          self.compiler = options[i+1]
        break

  def findIncludeDirs(self):
    if self.compiler:
      tail = os.path.split(self.compiler)[1]
      if tail.startswith('mpi'):
        # Check with '-show'
        cmd = self.compiler + ' -show'
        output = self.execute(cmd)
        if output:
          for op in output.split():
            if op.startswith('-I'):
              self.includeDir.append(op)
          return
        
        # Check with '--showme:incdirs'
        cmd = self.compiler + ' --showme:incdirs'
        output = self.execute(cmd)
        if output:
          self.includeDir.append('-I'+output.strip())
          return 


  def getIncludeDirs(self):
    self.findIncludeDirs()
    return ' '.join(self.includeDir)

  def execute(self, cmd):
    try:
      cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
      return cmdOutput.decode('utf-8')
    except subprocess.CalledProcessError as e:
      pass
    return None    

if __name__ == '__main__':
  cmd = sys.argv[1]
  mpi = MPIEnvironment(cmd)
  print(cmd)
  print(mpi.getIncludeDirs())
