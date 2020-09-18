import re

# Boolean options do not have an argument; they are either specified on a command line or not
# Format: (option) : (new_option)
BOOLEAN_OPTIONS = {
  '--verbose': '',
  '-v': '',
  '--device-c': '-fcuda-rdc', 
  '-dc': '-fcuda-rdc',
  '--cuda': '',
  '-cuda': '',
  '--cubin': '',
  '-cubin': '',
  '--fatbin': '',
  '-fatbin': '',
  '--ptx': '',
  '-ptx': '',
  '--preprocess': '',
  '-E': '',
  '--generate-dependencies': '',
  '-M': '',
  '--compile': '-c',
  '-c': '-c',
  '--device-w': '',
  '-dw': '',
  '--device-link': '',
  '-dlink': '',
  '--link': '',
  '-link': '',
  '--lib': '',
  '-lib': '',
  '--run': '',
  '-run': '',
  '--profile': '',
  '-pg': '',
  '--debug': '-g',
  '-g': '-g',
  '--device-debug': '',
  '-G': '',
  '--generate-line-info': '',
  '-lineinfo': '',
  '--shared': '',
  '-shared': '',
  '--no-host-device-initializer-list': '',
  '-nohdinitlist': '',
  '--no-host-device-move-forward': '-nohdmoveforward',
  '--expt-relaxed-constexpr': '',
  '--expt-extended-lambda': '',
  '--extended-lambda': '',
  '-extended-lambda': '',
  '--dont-use-profile': '',
  '-noprof': '',
  '--dryrun': '',
  '-dryrun': '',
  '--keep': '',
  '-keep': '',
  '--save-temps': '',
  '-save-temps': '',
  '--clean-targets': '',
  '-clean': '',
  '--no-align-double': '',
  '--no-device-link': '',
  '--use_fast_math': '',
  '-use_fast_math': '',
  '--disable-warnings': '',
  '-w': '',
  '--keep-device-functions': '',
  '-keep-device-functions': '',
  '--source-in-ptx': '',
  '-source-in-ptx': '',
  '--restrict': '',
  '-restrict': '',
  '--Wreorder': '',
  '-Wreorder': '',
  '--Wno-deprecated-declarations': '',
  '-Wno-deprecated-declarations': '',
  '--Wno-deprecated-gpu-targets': '',
  '-Wno-deprecated-gpu-targets': '',
  '--resource-usage': '',
  '--res-usage': '',
  '--help': '',
  '-h': '',
  '--version': '--version',
  '-V': '--version'
}

# Single value options must be specified at most once
# Format:  (option) : (new_option, new_value)
SINGLE_VALUE_OPTIONS = {
  '--output-file': ('-o', 'SAME'),
  '-o': ('-o', 'SAME'),
  ' --compiler-bindir': ('', '')
  '-ccbin': ('', '')
  '--output-directory': ('', '')
  '-odir': ('', '')
  '--cudart': ('', ''),
  '-cudart': ('', ''),
  '--libdevice-directory': ('', ''),
  '-ldir': ('', ''),
  '--use-local-env': ('', ''),
  '--optimize': ('-O', ''),
  '-O': ('-O', ''),
  '--ftemplate-backtrace-limit': ('', ''),
  '-ftemplate-backtrace-limit': ('', ''),
  '--ftemplate-depth': ('', ''),
  '--x': ('CHECK', 'CHECK'), ######################! cu should be cuda
  '-x': ('CHECK', 'CHECK'),
  '--std': ('-std', 'SAME'), ######################## should be -std=XXX
  '-std': ('-std', 'SAME'),
  '--machine': ('', ''),
  '-m': ('', ''),
  '--keep-dir': ('', ''),
  '-keep-dir': ('', ''),
  '--input-drive-prefix': ('', ''),
  '-idp': ('', ''),
  '--dependency-drive-prefix': ('', ''),
  '-ddp': ('', ''),
  '--drive-prefix': ('', ''),
  '-dp': ('', ''),
  '--dependency-target-name': ('', ''),
  '-MT': ('', ''),
  '--gpu-architecture': ('--cuda-gpu-arch', 'SAME'),
  '-arch': ('--cuda-gpu-arch', 'SAME'),
  '--generate-code': ('--cuda-gpu-arch', 'SAME'),
  '--gencode': ('--cuda-gpu-arch', 'SAME'),
  '--relocatable-device-code': ('CHECK', 'CHECK'), ################ if this false it should be -fno-cuda-rdc
  '-rdc': ('CHECK', 'CHECK'), ################ if this false it should be -fno-cuda-rdc
  '--maxrregcount': ('', ''),
  '-maxrregcount': ('', ''),
  '--ftz': ('', ''),
  '-ftz': ('', ''),
  '--prec-div': ('', ''),
  '-prec-div': ('', ''),
  '--prec-sqrt': ('', ''),
  '-prec-sqrt': ('', ''),
  '--fmad': ('', ''),
  '-fmad': ('', ''),
  '--default-stream': ('', ''),
  '-default-stream': ('', '')
}

# Tables for the CHECK case in the SINGLE_VALUE_OPTIONS table
SINGLE_VALUE_OPTIONS_TRANS = {
  '--x cu': '-x cuda',
  '--x c': '-x c',
  '--x c++': '-x c++',
  '-x cu': '-x cuda',
  '-x c': '-x c',
  '-x c++': '-x c++',
  '--relocatable-device-code true': '-fcuda-rdc',
  '--relocatable-device-code false': '-fno-cuda-rdc',
  '-rdc true': '-fcuda-rdc',
  '-rdc false': '-fno-cuda-rdc',
}

# List options may be repeated.
# Format: (option) : (new_option, new_value)
LIST_OPTIONS = {
  '--include-path': ('-I', 'SAME'),
  '-I': ('-I', 'SAME'), 
  '--ptxas-options': ('',''), 
  '-Xptxas':('',''),
  '--pre-include': ('-include', 'SAME'),
  '-include': ('-include', 'SAME'),
  '--library': ('', ''),
  '-l': ('', ''),
  '--define-macro': ('-D', 'SAME'),
  '-D': ('-D', 'SAME'),
  '--undefine-macro': ('-U', 'SAME'),
  '-U': ('-U', 'SAME'),
  '--system-include': ('-isystem', 'SAME'),
  '-isystem': ('-isystem', 'SAME'),
  '--library-path': ('-L', 'SAME'),
  '-L': ('-L', 'SAME'),
  '--compiler-options': ('', ''),
  '-Xcompiler': ('', ''),
  '--linker-options': ('', ''),
  '-Xlinker': ('', ''),
  '--archive-options': ('', ''),
  '-Xarchive': ('', ''),
  '--nvlink-options': ('', ''),
  '-Xnvlink': ('', ''),
  '--run-args': ('', ''),
  '-run-args': ('', ''),
  '--gpu-code': ('--cuda-gpu-arch', 'SAME'),
  '-code': ('--cuda-gpu-arch', 'SAME'),
  '--entries': ('', ''),
  '-e': ('', ''),
  '--Werror': ('-Werror', ''),
  '-Werror': ('-Werror', ''),
  '--options-file': ('', ''),
  '-optf': ('', '')
}

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
    return ' '.join(self.newCommand)

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
    #p = self.option.split('=')
    if '=' in self.option:
      opt = p[0]
      self.consumedTokens = 1
    else:
      opt = self.option
      self.consumedTokens = 2

    val = SINGLE_VALUE_OPTIONS[opt]
    self.newOption.append(val[0])

    if val[1] == 'SAME':
      if '=' in self.option:
        self.newOption.append(self.option.split('=')[1])
      else:
        self.newOption.append(self.tokens[self.idx+1])
    else:
      self.newOption.append(val[1])

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

    print('self.consumedTokens', self.consumedTokens)

if __name__ == '__main__':
  #cmd = ';cd dir'
  cmd = 'cd dir; nvcc -dc --verbose -o file.o --ptxas-options=-v,O2 -I/path/include --include-path /path1 -I /this/path -unknown -ignore -dc; nvcc -c file --verbose'
  c = ClangCommand(cmd)
  print(cmd)
  print(c.to_str())
  #tokens = cmd.split()
  #idx = 0
  #clangOpt = ClangOption.create(tokens[idx], tokens, idx)
  #help(clangOpt)
  #print(clangOpt.consumed())
  #print(clangOpt.to_str())
