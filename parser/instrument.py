
import os
import tempfile
import sys
from collections import defaultdict
from tokenizer import Tokenizer
from match import Match
from deprocess import Deprocess

## Assumes we already pre-processed the file
class Instrument:
  def __init__(self, preprocessedFile, srcFile):
    self.preFileContent = None
    self.preFileName = preprocessedFile
    self.sourceFileName = srcFile
    self.deprocessedFile = None
    self.deviceDclLines = []
    self.allTokens = []
    self.linesOfAssigments = defaultdict(list)
    self.transformedLines = {}

  def __del__(self):
    if self.deprocessedFile:
      os.remove(self.deprocessedFile)

  def deprocess(self):
    tmpFd, tmpFname = tempfile.mkstemp(suffix='.txt', text=True)
    print('Temp file:', tmpFname)
    self.deprocessedFile = tmpFname
    dp = Deprocess(self.preFileName, tmpFname)
    print('Running de-processor...')
    dp.run()
    #os.close(tmpFd)
    #os.remove(tmpFname)

  def findDeviceDeclarations(self):
    t = Tokenizer(self.deprocessedFile)
    for token in t.tokenize():
      self.allTokens.append(token)
    m = Match()
    self.deviceDclLines = m.match_device_function(self.allTokens)

  ## Finds ranges of lines that contain assigments
  def findAssigments(self):
    for l in self.deviceDclLines:
      startLine, endLine, startIndex, endIndex = l # unpack lines and indexes
      m = Match()
      tokenIndexes = m.match_assigment(self.allTokens[startIndex:endIndex])
      for t in tokenIndexes:
        i_abs = startIndex + t[0]
        j_abs = startIndex + t[1]
        i_line = self.allTokens[i_abs].lineNumber()
        j_line = self.allTokens[j_abs].lineNumber()
        self.linesOfAssigments[i_line].append((i_abs, 'b'))
        self.linesOfAssigments[j_line].append((j_abs, 'e'))
        print('Lines with assigments:', self.linesOfAssigments)

  def transformLine(self, index, currentLine):
    beg_tokens = set([])
    end_tokens = set([])
    for elem in self.linesOfAssigments[currentLine]:
      i, kind = elem
      if kind=='b': beg_tokens.add(i)
      if kind=='e': end_tokens.add(i)

    i = index
    newLine = ''
    for token in self.allTokens[index:]:
      if i in beg_tokens:
        newLine += 'PRE('+str(self.allTokens[i])
      elif i in end_tokens:
        newLine += ')END'+str(self.allTokens[i])
      else:
        newLine += str(self.allTokens[i])
      if str(token)=='\n':
        break
      i += 1

    return i-index, newLine 

  def produceInstrumentedLines(self):
    currentLine = 1
    index = -1
    while True:
      index += 1
      if index >= len(self.allTokens):
        break

      token = self.allTokens[index]
      if str(token)=='\n':
        currentLine += 1
        continue

      if currentLine in self.linesOfAssigments.keys():
        tokensConsumed, newLine = self.transformLine(index, currentLine)
        index += tokensConsumed-1
        print('==>', newLine)
        self.transformedLines[currentLine] = newLine

  def instrument(self):
    with open(self.sourceFileName, 'r') as fd:
      l = 0
      for line in fd:
        l += 1
        if l in self.transformedLines.keys():
          newLine = self.transformedLines[l]
          print(newLine[:-1])
        else:
          print(line[:-1])

if __name__ == '__main__':
  preFileName = sys.argv[1]
  sourceFileName = sys.argv[2]
  inst = Instrument(preFileName, sourceFileName)
  #inst.loadProcessedFile()
  inst.deprocess()
  inst.findDeviceDeclarations()
  print(inst.deviceDclLines)
  inst.findAssigments()
  inst.produceInstrumentedLines()
  inst.instrument()
  #inst.getContentOfAssigmentLines()
  #print(inst.assigmentIndexes)
  #inst.printInstrumentedCode()

