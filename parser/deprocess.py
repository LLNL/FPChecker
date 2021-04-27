
import os
import sys
from exceptions import DepreprocessorException

class Deprocess():
  ''' Remove content added by preprocessor. '''
  def __init__(self, fileName, newfile):
    self.fileName = os.path.abspath(fileName)
    self.newFileName = os.path.abspath(newfile)
    self.addedLines = 0

  def run(self):
    sourceFile = None
    inCurrentFile = False
    returningLine = 0

    with open(self.fileName, 'r') as orig:
      with open(self.newFileName, 'w') as dest:
        c = 0
        for line in orig:
          c += 1

          ## Get the original name of the file
          if c==1:
            if not line.startswith('# '):
              raise DepreprocessorException('Error: file has not been pre-processed: '+self.fileName)
            else:
              sourceFile = line.split()[2]

          ## Check if it's a pre-processor line
          if line.startswith('# '):
            if line.split()[2] == sourceFile:
              inCurrentFile = True
              _, returnningLine, *rest = line.split()
              returnningLine = int(returnningLine)
              self.addEmptyLines(dest, returnningLine-1)
            else:
              inCurrentFile = False

          ## Add content to new file
          if inCurrentFile:
            if not line.startswith('# '):
              dest.write(line)
              self.addedLines += 1

  def addEmptyLines(self, fd, finalLine):
    if finalLine > self.addedLines:
      diff = finalLine - self.addedLines
      for i in range(diff):
        fd.write('\n')
      self.addedLines += diff

if __name__ == '__main__':
  fileName = sys.argv[1]
  newFile = 'newFile.cpp'
  dp = Deprocess(fileName, newFile)
  print('Running de-processor...')
  dp.run()
