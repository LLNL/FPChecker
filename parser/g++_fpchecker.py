#!/usr/bin/env python3

import os
import pathlib
import subprocess
import sys
from colors import prGreen, prCyan, prRed
from instrument import Instrument
from exceptions import CommandException, CompileException, EmptyFileException
from fpc_logging import logMessage, verbose

# --------------------------------------------------------------------------- #
# --- Installation Paths ---------------------------------------------------- #
# --------------------------------------------------------------------------- #

# Main installation path
FPCHECKER_PATH      = str(pathlib.Path(__file__).parent.absolute())
FPCHECKER_RUNTIME   = FPCHECKER_PATH+'/../src/Runtime_g++.h'

#
# --------------------------------------------------------------------------- #
# --- Global Variables ------------------------------------------------------ #
# --------------------------------------------------------------------------- #

# File extensions that can have CUDA code
CPP_EXTENSION = ['.C', '.cc', '.cpp', '.CPP', '.c++', '.cp', '.cxx']
COMPILER_NAME = 'g++'

# --------------------------------------------------------------------------- #
# --- Classes --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

class Command:
  def __init__(self, cmd):
    self.name = cmd[0]
    self.parameters = cmd[1:]
    self.preprocessedFile = None
    self.instrumentedFile = None
    self.outputFile = None

  def executeOriginalCommand(self):
    try:
      cmd = [COMPILER_NAME] + self.parameters
      if verbose(): print('Executing original command:', cmd)
      subprocess.run(' '.join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
      prRed(e)

  def getOriginalCommand(self):
    return ' '.join([COMPILER_NAME] + self.parameters[1:])

  # It it a compilation command?
  def isCompileCommand(self) -> bool:
    if ('-c' in self.parameters or
        '--compile' in self.parameters):
      return True
    return False

  # Is the command a link command?
  def isLinkCommand(self) -> bool:
    if ('-c' not in self.parameters and 
        '--compile' not in self.parameters and
        ('-o' in self.parameters or '--output-file' in self.parameters)):
      return True
    return False

  # Get the name of the cuda file to be compiled
  # if the file exists.
  def getCodeFileNameIfExists(self):
    global CPP_EXTENSION
    fileName = None
    for t in self.parameters:
      for ext in CPP_EXTENSION:
        if t.endswith(ext):
          fileName = t

    if not fileName:
      message = 'Could not find source file in command'
      logMessage(message)
      raise CommandException(message)
  
    return fileName

  def getOutputFileIfExists(self):
    for i in range(len(self.parameters)):
      p = self.parameters[i]
      if p == '-o' or p == '--output-file':
        self.outputFile = self.parameters[i+1]
        return self.parameters[i+1]
    return None

  # We transform the command and execute the pre-proecessor
  def executePreprocessor(self):
    source = self.getCodeFileNameIfExists()
 
    # Copy the command parameters
    newParams = self.parameters.copy()

    outputFile = self.getOutputFileIfExists()
    if outputFile:
      self.preprocessedFile = outputFile + '.ii'
      for i in range(len(newParams)):
        p = self.parameters[i]
        if p == '-o' or p == '--output-file':
          newParams[i+1] = self.preprocessedFile
          break
    else:
      self.preprocessedFile = source + '.ii'
      newParams.append('-o')
      newParams.append(self.preprocessedFile)
    
    new_cmd = [COMPILER_NAME, '-E'] + newParams
    try:
      if verbose(): prGreen(' '.join(new_cmd)) 
      cmdOutput = subprocess.run(' '.join(new_cmd), shell=True, check=True)
    except Exception as e:
      message = 'Could not execute pre-processor'
      if verbose():
        prRed(e)
        logMessage(str(e))
        logMessage(message)
      raise RuntimeError(message) from e

    return True

  def instrumentSource(self):
    preFileName = self.preprocessedFile
    sourceFileName = self.getCodeFileNameIfExists()
    inst = Instrument(preFileName, sourceFileName)
    inst.deprocess()
    #inst.findDeviceDeclarations()
    inst.findAllDeclarations()
    if verbose(): print(inst.deviceDclLines)
    inst.findAssigments()
    inst.produceInstrumentedLines()
    inst.instrument()
    self.instrumentedFile = inst.getInstrumentedFileName()

  def compileInstrumentedFile(self):
    source = self.getCodeFileNameIfExists()
    # Copy original command
    new_cmd = [COMPILER_NAME, '-include', FPCHECKER_RUNTIME] + self.parameters
    # Replace file by instrumented file
    for i in range(len(new_cmd)):
      p = new_cmd[i]
      if p == source:
        new_cmd[i] = self.instrumentedFile
        break

    # Change output file
    if not self.outputFile:
      fileName, ext = os.path.splitext(source)
      newOutputFile = fileName + '.o'
      new_cmd = new_cmd + ['-o', newOutputFile]

    # Compile
    try:
      if verbose(): prGreen('Compiling: ' + ' '.join(new_cmd))
      cmdOutput = subprocess.run(' '.join(new_cmd), shell=True, check=True)
    except Exception as e:
      if verbose():
        prRed(e)
        logMessage(str(e))
        message = 'Could not compile instrumented file'
        logMessage(message)
      raise CompileException(message) from e

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
      cmd.executePreprocessor()
      cmd.instrumentSource()
      cmd.compileInstrumentedFile()
      logMessage('Instrumented: ' + cmd.instrumentedFile)
    except Exception as e: # Fall back to original command
      if verbose():
        logMessage(str(e))
        prRed(e)     
      if not isinstance(e, EmptyFileException):
        logMessage('Failed: ' + ' '.join(sys.argv))
      else:
          if verbose():
            logMessage('Failed: ' + ' '.join(sys.argv))
      cmd.executeOriginalCommand()




