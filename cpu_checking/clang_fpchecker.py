#!/usr/bin/env python3

# Description: This script replaces the clang or clang++ command.
#              It adds the flags required to load the LLVM pass
#              and include the runtime header file.

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
# --- Classes --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Command:
  def __init__(self, cmd):
    # We assume the command is clang-fpchecker or clang++-fpchecker
    if os.path.split(cmd[0])[1].split('-')[0].endswith('clang++'):
      self.name = 'clang++'
    else:
      self.name = 'clang'
    self.parameters = cmd[1:]
    self.preprocessedFile = None
    self.instrumentedFile = None
    self.outputFile = None

  def executeOriginalCommand(self):
    try:
      cmd = [self.name] + self.parameters
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

  def getOutputFileIfExists(self):
    for i in range(len(self.parameters)):
      p = self.parameters[i]
      if p == '-o' or p == '--output-file':
        self.outputFile = self.parameters[i+1]
        return self.parameters[i+1]
    return None

  def instrumentIR(self):
    new_cmd = [self.name] + LLVM_PASS.split() + self.parameters
    for p in self.parameters:
      if '-fopenmp' in p:
        new_cmd += ['-DFPC_FPC_MULTI_THREADED']
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

