import ast
import os
import sys
import glob
from colors import prGreen,prCyan,prRed

TRACES_DIR = './.fpchecker/traces'
TRACES_FILES = TRACES_DIR+'/'+'trace'


class TraceBackCommand:

  def __init__(self):
    pass

  @staticmethod
  def getFile(rawCommand):
    files = glob.glob(TRACES_DIR+'/trace.*')
    fileName = None
    for f in files:
      with open(f) as fd:
        for line in fd:
          if rawCommand in line:
            fileName =  f
            break
    return fileName       

  @staticmethod
  def checkDirectory():
    if not os.path.isdir(TRACES_DIR):
      raise SystemExit('Error: traces directory does not exist: '+TRACES_DIR)
    if not os.listdir(TRACES_DIR):
      raise SystemExit('Error: '+TRACES_DIR+' is empty') 

  @staticmethod  
  def printTrace(lineNumber: int):
    TraceBackCommand.checkDirectory()

    if int(lineNumber) < 1:
      raise SystemExit('Error: command ID must be >= 1')

    lineNumber = int(lineNumber)
    prGreen('Recreating traces for command: ' + str(lineNumber))

    tracesFile = TRACES_DIR+'/raw_traces.txt'
    if not os.path.isfile(tracesFile):
      raise SystemExit('Error '+ tracesFile + ' does not exist. Please record build traces again.')

    trace = None
    with open(tracesFile, 'r') as fd:
      i = 0
      for line in fd:
        i += 1
        if i == lineNumber:
          trace = ast.literal_eval(line)[1]

    if trace:
      print('Low-level command:')
      print(trace.strip()+'\n')
      print('It may take a few minutes...')
      fileName = TraceBackCommand.getFile(trace)
      #print(fileName)
      if fileName:
        TraceBackCommand.printTree(fileName)
    else:
      raise SystemExit('Error: could not find the command in the trace file')

  @staticmethod
  def getParent(fileName):
    fileName = os.path.split(fileName)[1]
    PID = fileName.split('.')[1]
    files = glob.glob(TRACES_DIR+'/trace.*')
    parent = None
    for f in files:
      with open(f) as fd:
        for line in fd:
          if 'fork(' in line or 'clone(' in line:
            if line.endswith('= '+PID+'\n'):
              parent = f
              #print('found parent:', parent)
              break
      if parent:
        break

    if parent:
      return parent
    else:
      sys.exit()

  @staticmethod
  def printTree(fileName):
    if fileName == None:
      return

    print('<Command>')
    execCmd = None
    with open(fileName, 'r') as fd:
      for line in fd:
        if 'execve(' in line:
          #print(line)
          returnValue = line.split('=')[-1:][0].strip()
          #print('returnValue', returnValue)
          if returnValue == '0':
            execCmd = line
            print('\t'+execCmd.strip())
            print('\t'+fileName)
            parentFile = TraceBackCommand.getParent(fileName)
            TraceBackCommand.printTree(parentFile)

  @staticmethod
  def getSomeCommands():
    files = glob.glob(TRACES_DIR+'/trace.*')
    fileName = None
    maxCount = 13
    traces = []
    i = 0
    for f in files:
      with open(f) as fd:
        for line in fd:
          if 'execve(' in line and 'bin2c' in line:
            traces.append(('', line))
            i += 1
            print('i:', i)
      if i == maxCount:
        break
    with open('new_traces.txt', 'w') as fd:
      for l in traces:
        fd.write(str(l)+'\n')

if __name__ == '__main__':
  #cmdId = sys.argv[1]
  #TraceBackCommand.printTrace(cmdId)
  TraceBackCommand.getSomeCommands()
