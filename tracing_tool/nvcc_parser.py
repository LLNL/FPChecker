import re
import sys
from nvcc_options_table import BOOLEAN_OPTIONS, SINGLE_VALUE_OPTIONS, SINGLE_VALUE_OPTIONS_TRANS, LIST_OPTIONS  

class ClangCommand:

  def __init__(self, line):
    # Split into multiple commands
    #newCommand = []
    self.newCommand = []
    allCommands = re.split('\&\&|\;', line)
    for cmd in allCommands:
      if len(self.newCommand) > 0:
        self.newCommand.append(';')

      if len(cmd) == 0:
        continue

      newCommand = []
      tokens = cmd.split()
      idx = 0 # current index

      while True:
        elem = tokens[idx]
        clangOpt = ClangOption.create(tokens[idx], tokens, idx)
        newCommand.append(clangOpt.to_str())
        consumed = clangOpt.consumed()
        idx = idx + consumed
        if idx >= len(tokens):
          break

      self.newCommand = self.newCommand + newCommand

  def to_str(self):
    for elem in self.newCommand:
      if elem == 'nvcc' or elem.endswith('/nvcc'):
        idx = self.newCommand.index(elem)
        self.newCommand[idx] = 'clang++'
    
    ret = ' '.join(self.newCommand)
    #ret = ret.replace('nvcc ', 'clang++ ')

    # Add default compute capability if not present
    if '--cuda-gpu-arch' not in ret:
      ret = ret.replace('clang++ ', 'clang++ --cuda-gpu-arch=sm_60 ')
    return ret

class ClangOption:

  def __init__(self, opt, tok, i):
    self.option = opt
    self.tokens = tok
    self.idx = i
    self.consumedTokens = 1
    self.newOption = []

  @classmethod
  def create(cls, opt, tok, i):
    if '=' in opt:
      v = opt.split('=')[0]
    else:
      v = opt

    if v in BOOLEAN_OPTIONS:
      b = BooleanOption(opt, tok, i)
      return b
    elif v in SINGLE_VALUE_OPTIONS:
      b = SingleValueOption(opt, tok, i)
      return b
    elif v in LIST_OPTIONS:
      b = ListOption(opt, tok, i)
      return b

    return ClangOption(opt,[],0)

  def to_str(self):
    ret = ' '.join(self.newOption)
    if len(self.newOption) == 0:
      ret = self.option
    return ret

  def consumed(self):
    return self.consumedTokens

class BooleanOption(ClangOption):
  def __init__(self, opt, tok, i):
    super().__init__(opt, tok, i)
    self.convertOption()

  def convertOption(self):
    val = BOOLEAN_OPTIONS[self.option]
    self.newOption.append(val)
    self.consumedTokens = 1

class SingleValueOption(ClangOption):
  def __init__(self, opt, tok, i):
    super().__init__(opt, tok, i)
    self.convertOption()

  def convertOption(self):
    if '=' in self.option:
      opt = self.option.split('=')[0]
      self.consumedTokens = 1
    else:
      opt = self.option
      self.consumedTokens = 2

    converted_option = []
    val = SINGLE_VALUE_OPTIONS[opt]

    if val[0] != 'CHECK': #---------------------------------
      #self.newOption.append(val[0])
      converted_option.append(val[0])

      if val[1] == 'SAME':
        if '=' in self.option:
          #self.newOption.append(self.option.split('=')[1])
          converted_option.append(self.option.split('=')[1])
        else:
          #self.newOption.append(self.tokens[self.idx+1])
          converted_option.append(self.tokens[self.idx+1])
      else:
        #self.newOption.append(val[1])
        converted_option.append(val[1])
    else: # CHECK value ------------------------------------
      whole_option = ''
      if '=' in self.option:
        p = self.option.split('=')
        whole_option = p[0]+' '+p[1]
      else:
        whole_option = self.option+' '+self.tokens[self.idx+1]
      newOpt = SINGLE_VALUE_OPTIONS_TRANS[whole_option]
      #self.newOption.append(newOpt)
      converted_option.append(newOpt)
    
    if val[2] == 1:
      self.newOption.append('='.join(converted_option))
    else:
      self.newOption.append(' '.join(converted_option))


class ListOption(ClangOption):
  def __init__(self, opt, tok, i):
    super().__init__(opt, tok, i)
    self.convertOption()

  def convertOption(self):
    #p = self.option.split('=')
    if '=' in self.option:
      opt = self.option.split('=')[0]
      self.consumedTokens = 1
    else:
      opt = self.option
      self.consumedTokens = 2

    val = LIST_OPTIONS[opt]
    self.newOption.append(val[0])

    if val[1] == 'SAME':
      if '=' in self.option:
        self.newOption.append(self.option.split('=')[1])
      else:
        self.newOption.append(self.tokens[self.idx+1])
    else:
      self.newOption.append(val[1])

if __name__ == '__main__':
  cmd = ' '.join(sys.argv[1:])
  #print('Convert:', cmd)
  #cmd = 'nvcc -c -rdc false -rdc=true --x cu -x=cu -std c++ -std=c -arch sm_60'
  #cmd = 'cd dir; nvcc -dc --verbose -o file.o --ptxas-options=-v,O2 -I/path/include --include-path /path1 -I /this/path -unknown -ignore -dc; nvcc -c file --verbose'
  c = ClangCommand(cmd)
  print(c.to_str())
