#!/usr/bin/env python3

import tempfile
import shutil
import os
import sys
import re
from collections import deque

expressionPattern_EQ      = re.compile(r"(.*?)([a-zA-Z0-9_\[\]\s]+)(\+?\=)([a-zA-Z0-9_\.\[\]\(\)\+\-\*\/\n\s]+);[\s]*\n", re.DOTALL)

def representsInt(s):
  try: 
    int(s)
    return True
  except ValueError:
    return False

class ParseExpressions:

  windowSize = 10
  modifiedFile = []
  fileName = ''

  def __init__(self, fileName):

    self.fileName = fileName
    previous_lines = deque(maxlen=self.windowSize)
    line_numbers = deque(maxlen=self.windowSize)
    with open(fileName, 'r') as fd:
      number = 0
      for line in fd:
        number += 1
        previous_lines.append(line)
        line_numbers.append(number)
        if len(previous_lines) == self.windowSize:
          linesMatched = self.parseWindow(previous_lines, line_numbers)
          for i in range(linesMatched):
            if len(previous_lines) > 0:
              previous_lines.popleft()
              line_numbers.popleft()

    #print('Final QUEUE:\n', previous_lines)
    self.saveFinalFile(previous_lines)

  def saveFinalFile(self, previous_lines):
    tmpFd, tmpFname = tempfile.mkstemp(suffix='.txt', text=True)
    with open(tmpFname, 'w') as f:
      for l in self.modifiedFile:
        f.write(l)
      for l in list(previous_lines):
        f.write(l)
    os.close(tmpFd)

    # Copy tmp file to original file
    shutil.copy2(tmpFname, self.fileName)
    # Remove tmp file
    os.remove(tmpFname)

  def consumedLines(self, win, expression):
    wholeWindow = ''.join(list(win))
    idx = wholeWindow.find(expression)
    lines = wholeWindow[:idx+len(expression)+1].count('\n')
    return lines

  def matchPattern(self, fullLine):
    foundPattern = expressionPattern_EQ.search(fullLine)
    if foundPattern:
       return foundPattern
    return None

  def RHSIsValid(self, RHS):
    if (not representsInt(RHS) and
        'default' not in RHS and
        'NULL' not in RHS and 'nullptr' not in RHS and
        'true' not in RHS and 'false' not in RHS and 
        '//' not in RHS and '/*' not in RHS and '*/' not in RHS):
      return True
    return False
  
  def LHSIsValid(self, PRE, LHS):
    if ('&' not in PRE+LHS):
      return True
    return False

  def parseWindow(self, win, line_numbers):
    fullLine = ''.join(list(win))
    foundPattern = self.matchPattern(fullLine)
    #print("\nWINDOW:", fullLine)
    if foundPattern:
      block = foundPattern.group(0)
      expression = foundPattern.group(2) + foundPattern.group(3)

      if ('{' not in expression and
          '}' not in expression and
          'using' not in expression):

        PRE = foundPattern.group(1) 
        LHS = foundPattern.group(2)
        EQS = foundPattern.group(3) # Equal sign
        RHS = foundPattern.group(4)
  
        if self.RHSIsValid(RHS) and self.LHSIsValid(PRE, LHS):
          consumed = self.consumedLines(win, block)
          lineNumber = str(list(line_numbers)[consumed-1])
          # Example: double _FPC_CHECK_(double x, int loc, const char *fileName)
          newLine = PRE + LHS + EQS + ' _FPC_CHECK_MACRO_(' + RHS + ', ' + lineNumber + ', \"' + self.fileName[-25:] + '..."); \n'
          self.modifiedFile.append(newLine)
          return consumed 

    self.modifiedFile.append(fullLine)
    return self.windowSize

if __name__ == "__main__":
  fileName = sys.argv[1]
  print('Transforming file:', fileName+'...')
  exp = ParseExpressions(fileName)
