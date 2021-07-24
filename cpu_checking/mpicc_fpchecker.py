#!/usr/bin/env python3

import os
import pathlib
import subprocess
import platform
import sys
from colors import prGreen, prCyan, prRed
from exceptions import CommandException, CompileException, EmptyFileException
from fpc_logging import logMessage, verbose

# --------------------------------------------------------------------------- #
# --- Installation Paths ---------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Main installation path
FPCHECKER_PATH      = str(pathlib.Path(__file__).parent.absolute())
if platform.system() == 'Darwin':
  FPCHECKER_LIB       = FPCHECKER_PATH+'/../lib/libfpchecker_cpu.dylib'
else:
  FPCHECKER_LIB       = FPCHECKER_PATH+'/../lib/libfpchecker_cpu.so'
FPCHECKER_RUNTIME   = FPCHECKER_PATH+'/../src/Runtime_cpu.h'
LLVM_PASS           = "-Xclang -load -Xclang " + FPCHECKER_LIB + " -include " + FPCHECKER_RUNTIME + ' -g '

# --------------------------------------------------------------------------- #
# --- Global variables ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

C_WRAPPER_NAMES   = ['mpicc', 'mpiclang', 'mpigcc']
CXX_WRAPPER_NAMES = ['mpiCC', 'mpic++', 'mpicxx', 'mpiclang++', 'mpig++']

# --------------------------------------------------------------------------- #
# --- Classes --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Command:
  def __init__(self, cmd):
    # We assume the command is NAME-fpchecker, where NAME is one of the
    # supported C or CXX wrapper names
    wrapper_name = os.path.split(cmd[0])[1].split('-')[0]
    if (wrapper_name in C_WRAPPER_NAMES):
      self.name = "clang"
    elif (wrapper_name in CXX_WRAPPER_NAMES):
      self.name = "clang++"
    else:
      raise CompileException("Invalid MPI wrapper name")
    self.wrapper_name = wrapper_name
    self.parameters = cmd[1:]
    self.mpi_params = self.getMPICompileParams()

  def getMPICompileParams(self):
    cmd = self.wrapper_name + ' --showme:compile'
    try:
      cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
      params = cmdOutput.decode('utf-8')
      ret = []
      for i in params.split():
        if 'rpath' not in i:
          ret.append(i)
      return ret
    except subprocess.CalledProcessError as e:
      prRed(e)

  def executeOriginalCommand(self):
    try:
      cmd = [self.wrapper_name] + self.parameters
      if verbose(): print('Executing original command:', ' '.join(cmd))
      subprocess.run(' '.join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
      prRed(e)

  def getOriginalCommand(self):
    return ' '.join([self.name] + self.parameters[1:])

  # It it a compilation command?
  def isCompileCommand(self) -> bool:
    if ('-c' in self.parameters):
      return True
    return False

  # Is the command a link command?
  def isLinkCommand(self) -> bool:
    if ('-c' not in self.parameters and 
        '--compile' not in self.parameters and
        ('-o' in self.parameters or '--output-file' in self.parameters)):
      return True
    return False

  def instrumentIR(self):
    new_cmd = [self.name] + self.mpi_params + LLVM_PASS.split() + self.parameters
    try:
      cmdOutput = subprocess.run(' '.join(new_cmd), shell=True, check=True)
    except Exception as e:
      prRed(e)
      raise CompileException(new_cmd) from e

if __name__ == '__main__':
  cmd = Command(sys.argv)

  if 'FPC_INSTRUMENT' not in os.environ:
    cmd.executeOriginalCommand()
    exit()

  # Link command
  if cmd.isLinkCommand():
    cmd.executeOriginalCommand()
  else:
    # Compilation command
    try:
      cmd.instrumentIR()
    except Exception as e: # Fall back to original command
      logMessage(str(e))
      prRed(e)     
      logMessage('Failed: ' + ' '.join(sys.argv))
      cmd.executeOriginalCommand()

