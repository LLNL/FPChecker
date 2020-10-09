import os
import shutil
import subprocess
import sys
import re
import glob
from colors import prGreen,prCyan,prRed

TRACES_DIR = './.fpchecker/traces'
TRACES_FILES = TRACES_DIR+'/'+'trace'

STRACE = 'strace'

SUPPORTED_COMPILERS = set([
  'nvcc',
  'c++',
  'cc',
  'gcc',
  'g++',
  'xlc', 
  'xlC', 
  'xlc++', 
  'xlc_r', 
  'xlc++_r',
  'mpic',
  'mpic++',
  'mpicxx',
  'mpicc',
  'mpixlc',
  'mpixlC',
  'mpixlf',
  'mpif77',
  'mpif90',
  'clang',
  'clang++',
  'gfortran',
  'xlf',
  'xlf-gpu',
  'xlf2003',
  'xlf2003-gpu',
  'xlf2003_r',
  'xlf2003_r-gpu',
  'xlf2008',
  'xlf2008-gpu',
  'xlf2008_r',
  'xlf2008_r-gpu',
  'xlf90',
  'xlf90-gpu',
  'xlf90_r',
  'xlf90_r-gpu',
  'xlf95',
  'xlf95-gpu',
  'xlf95_r',
  'xlf95_r-gpu',
  'xlf_r',
  'xlf_r-gpu'
])

SUPPORTED_TOOLS = set([
  'ar',
  'ranlib',
  'bin2c'
])

# Examples of top commands
# [pid 83362] execve("/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc", 
# [pid 63885] execve("/bin/sh", ["/bin/sh", "-c", "cd /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/build/src/util && /usr/tcetmp/bin/c++     -o CMakeFiles/util.dir/util.cpp.o -c /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/src/util/util.cpp"]

# Saves Compilation commands
class CommandsTracing:

  #open("/usr/tcetmp/packages/spack/opt/spack/linux-redhat7-ppc64le/gcc-4.8.5/gcc-4.9.3-3clrxj5wz2i54h
  #[pid  8690] execve("/usr/tcetmp/bin/c++", ["/usr/tcetmp/bin/c++", "CMakeFiles/main.dir/src/main.cpp.o", "-o", "main"]
  pidPattern = re.compile("^\[pid\s+[0-9]+\] ")

  # clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x200000044f60) = 55734
  # vfork()                                 = 55729
  childSpawn_clone = re.compile("^clone\(.+=\s+[0-9]+")
  childSpawn_fork = re.compile("^vfork\(\).+=\s+[0-9]+")

  # Chdir call
  # chdir("/usr/workspace/wsa/laguna/fpchecker/clang_tool/wrapper/apps/RAJA_perf/RAJAPerf/build_ilaguna_build/tpl/RAJA") = 0
  chdirPattern = re.compile("^chdir\(.+\s+=\s+[0-9]+")

  # Fork from root:
  # vfork(strace: Process 22625 attached
  # Other forks:
  # [pid 95927] stat("/usr/gapps/resmpi/llvm/ppc64le/llvm-openmp-trunk-install/lib/tls/power9/altivec", strace: Process 95932 attached
  # [pid 22631] vfork(strace: Process 22634 attached
  # [pid 78391] clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x200000044f60) = 78392
  # [pid 86430] clone(strace: Process 86431 attached
  #attachPattern1 = re.compile("vfork\(strace\:\s+Process\s+[0-9]+\s+attached")
  #attachPattern_clone = re.compile("clone\(.+=\s+[0-9]+")
  #attachPattern_attach = re.compile("Process\s+[0-9]+\s+attached")

  # Process creation patterns:
  # We trace vfork() and clone()
  #[pid 69813] clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x200000044f60) = 69814
  # [pid 129570] <... clone resumed>child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x200000044f60) = 129601
  #[pid 69807] <... vfork resumed>)        = 69808
  childCreationPattern_clone_1 = re.compile("^\[pid\s+[0-9]+\] clone\(.+=\s+[0-9]+")
  childCreationPattern_clone_2 = re.compile("^\[pid\s+[0-9]+\] \<\.\.\. clone resumed\>.+=\s+[0-9]+")
  childCreationPattern_fork = re.compile("^\[pid\s+[0-9]+\] .+vfork.+=\s+[0-9]+")

  readPattern = re.compile("^\[pid\s+[0-9]+\] read\(")
  writePattern = re.compile("^\[pid\s+[0-9]+\] write\(")

  def __init__(self, make_command):
    self.traced_commands = []
    self.make_command = make_command
    self.childTree = {}
    self.parentTree = {}
    self.tracedPIDs = set([])

  def getTracesDir(self):
    return TRACES_DIR

  def isChildSpawn(self, line):
    child_fork = self.childSpawn_fork.search(line)
    child_clone = self.childSpawn_clone.search(line)

    pid = None
    if child_fork != None or child_clone != None:
      pid = line.split()[-1:][0]

    return pid

  def isMakeCommand(self, line):
    ret = False
    if "execve(\"" in line:
      # execve("/usr/tcetmp/bin/make", ["make", "-j"], 0x7fffffffb780 /* 128 vars */) = 0
      cmd = line.split(', [')[1].split('], ')[0]
      cmd = cmd.replace('"','')
      cmd = cmd.replace(',','')
      cmd = cmd.split()
      #print(cmd, self.make_command)
      if cmd == self.make_command:
        return True
    return ret

  def getRootFile(self):
    # Find root file
    files = glob.glob(TRACES_DIR+'/trace.*')
    root_file = ''
    for f in files:
      #print('Checking', f)
      with open(f) as fd:
        first_line = fd.readline()
        if self.isMakeCommand(first_line):
          root_file = f
          break
    #print('Root file', root_file)
    if root_file == '':
      prRed('Error: root file not found')
      exit(-1)
    return root_file 

  # Check if it is a chdir() system call
  # chdir("/usr/workspace/wsa/laguna/fpchecker/clang_tool/wrapper/apps/RAJA_perf/RAJAPerf/build_ilaguna_build/tpl/RAJA") = 0
  def isChangeDir(self, line):
    chdir_found = self.chdirPattern.search(line)
    newDir = None
    if chdir_found != None:
      if line.split()[2] == '0': # check it ends with 0
        newDir = line
    return newDir

# Old implementation of recursive search
# It has a bug on the cwd (it's kept for any recent process)
# We want to unset the cwd once the process examination exits
#
#  def recursiveTreeTraversal(self, fileName):
#    with open(fileName) as fd:
#      for line in fd:
#        # Save current dir
#        cwd = self.isChangeDir(line)
#        if cwd != None:
#          print('Found chdir: ', cwd, 'file:', fileName)
#          self.currentWorkingDir = cwd
#
#        # Check if it's a top command, and it if so
#        topCmd = self.isTopCommand(line)
#        if topCmd != None:
#          # Add CWD and command
#          print('Adding:')
#          print('self.currentWorkingDir: ', self.currentWorkingDir)
#          print('line:', line)
#          self.traced_commands.append((self.currentWorkingDir, line))
#          return
#
#        # Check if child is created
#        childPID = self.isChildSpawn(line)
#        if childPID != None:
#          childFileName = TRACES_DIR + '/trace.' + childPID
#          self.recursiveTreeTraversal(childFileName)

  def recursiveTreeTraversal(self, fileName, chdirCmd):
    lastSeenCHDIR = chdirCmd
    with open(fileName) as fd:
      for line in fd:
        # Save current dir
        cwd = self.isChangeDir(line)
        if cwd != None:
          lastSeenCHDIR = cwd

        # Check if it's a top command, and it if so
        topCmd = self.isTopCommand(line)
        if topCmd != None:
          # Add CWD and command
          self.traced_commands.append((lastSeenCHDIR, line))
          return

        # Check if child is created
        childPID = self.isChildSpawn(line)
        if childPID != None:
          childFileName = TRACES_DIR + '/trace.' + childPID
          self.recursiveTreeTraversal(childFileName, lastSeenCHDIR)

  def analyzeTraces(self):
    #prveTreeTraversal(root_file)
    prGreen('Searching root PID...')
    root_file = self.getRootFile()
    print('root:', root_file)
    prGreen('Analyzing traces...')
    self.recursiveTreeTraversal(root_file, '')

  def getProcessID(self, line):
    p = self.pidPattern.match(line)
    #print('match', p)
    if p != None:
      pid = line.split()[1].split(']')[0]
    else:
      pid = 'root'
    return pid

  def buildChildTree(self, line):
    pid = self.getProcessID(line)
    child = None
    child_clone_1 = self.childCreationPattern_clone_1.search(line)
    child_clone_2 = self.childCreationPattern_clone_2.search(line)
    child_fork = self.childCreationPattern_fork.search(line)
    read_pattern = self.readPattern.search(line)
    write_pattern = self.writePattern.search(line)

    if child_clone_1 != None and read_pattern == None and write_pattern == None:
      child = line.split()[-1:][0]
    elif child_clone_2 != None and read_pattern == None and write_pattern == None:
      child = line.split()[-1:][0]
    elif child_fork != None and read_pattern == None and write_pattern == None:
      child = line.split()[-1:][0]
      
    if child != None: # found child creation
      if pid not in self.childTree:
        self.childTree[pid] = [child]
      else:
        self.childTree[pid].append(child)
      
      self.parentTree[child] = pid
      if pid in self.tracedPIDs:
        self.tracedPIDs.add(child)
 
  def isASupportedCompiler(self, line):
    for compiler in SUPPORTED_COMPILERS:
      if line.endswith('/'+compiler): #or line == compiler:
        return True

    for tool in SUPPORTED_TOOLS:
      if line.endswith('/'+tool):
        return True

    return False

  # If it's a top command we do not trace their child commands
  def isTopCommand(self, line):
    baseExecutable = None
    
    if "execve(\"" in line:
      strCmd = line.split('execve(')[1].split(',')[0]

      # Shell command
      if strCmd.endswith('/sh"'):
          cmd = line.split('["')[1].split(']')[0]
          cmd = cmd.replace(', ','')
          cmd = cmd.replace('"', '')
          tokens = cmd.split()
          for t in tokens:
            if self.isASupportedCompiler(t):
              baseExecutable = ' '.join(tokens)
      
      strCmd = strCmd.replace('"', '')
      if self.isASupportedCompiler(strCmd):
        baseExecutable = strCmd 

    return baseExecutable

  # [pid 78395] write(1, "[ 33%] Linking CXX static library libutil.a\n", 44[ 33%] Linking CXX static library libutil.a
  def printStdOut(self, line):
    if 'write(1' in line:
      if 'Building' in line or 'Linking' in line:
        l = line.split(', ')[1].replace('"','')
        prGreen(l)

  def saveCompilingCommands(self, l):
    #l = line.decode('utf-8')
    pid = self.getProcessID(l)
    cmd = self.isTopCommand(l)
    if cmd != None:
      if pid not in self.tracedPIDs:
        self.tracedPIDs.add(pid)
        self.traced_commands.append(l)
        #print('-->', cmd)

    self.buildChildTree(l)
    self.printStdOut(l)

  # Check if the command invokes chaning directories
  # If not, we change to the CWD
  def commandIvokesChangeDir(self, line):
    tokens = line.split()
    if 'cd' in tokens:
      idx = tokens.index('cd')
      path = tokens[idx+1]
      if os.path.exists(path):
        return True
    return False

  def formatCommandForExecution(self, cwd, line):
    if line.startswith('execve('):
      line = line.split(', [')[1:]
      line = ' '.join(line).split(']')[0]
      line = line.replace(', ',' ')
      line = line.replace('"', '')
      line = line.replace('\\', '')

      # Split commands if needed
      allCommands = re.split('\&\&|\;', line)
      newCommand = []
      for cmd in allCommands:
        if '/sh -c' in cmd:
          cmd = ' '.join(cmd.split()[2:]) # remove /bin/sh -c
        if '-E ' in cmd: # Remove commands that only run the preprocessor with -E
          continue
        if not self.commandIvokesChangeDir(line):
          cmd = 'cd ' + cwd + ' && ' + cmd
         
        newCommand.append(cmd) 

      line = ' && '.join(newCommand)

    return line

  def writeToFile(self):
    fileNameRaw = TRACES_DIR + '/raw_traces.txt'
    prGreen('Saving raw traces in '+fileNameRaw)
    fd = open(fileNameRaw, 'w')
    for line in self.traced_commands:
      fd.write(str(line)+'\n')
    fd.close()

    fileNameExec = TRACES_DIR + '/executable_traces.txt'
    prGreen('Saving executable traces in '+fileNameExec)
    fd = open(fileNameExec, 'w')

    for l in self.traced_commands:
      #line = l[1]
      cwd, line = l
      #if l[0] != '':
      #  cwd = l[0].split('"')[1]
      #else:
      #  cwd = '.'

      if cwd != '':
        cwd = cwd.split('"')[1]
      else:
        cwd = '.'
     
      line = self.formatCommandForExecution(cwd, line)

      fd.write(line+'\n')
    fd.close()

  def replayTraces(self, fileName):
    fd = open(fileName, 'r')
    for line in fd:
      self.saveCompilingCommands(line)
    fd.close()

  def createTracesDir(self):
    if os.path.exists(TRACES_DIR):
      shutil.rmtree(TRACES_DIR)
    os.makedirs(TRACES_DIR)

  def startTracing(self):
    self.createTracesDir()
    trace_command = [STRACE, '-o', TRACES_FILES, '-ff', '-s', '9999'] + self.make_command
    process = subprocess.Popen(trace_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Poll process for new output until finished
    c = 0
    while True:
      nextline = process.stdout.readline()
      #nextline = process.stderr.readline()
      if process.poll() is not None or nextline.decode('utf-8') == '':
        break

      l = nextline.decode('utf-8')[:-1]
      print(l)
      #self.saveCompilingCommands(l)
      #fd.write(l)

    (stdout_data, stderr_data) = process.communicate()
    exitCode = process.returncode

    if (exitCode == 0):
      return (stdout_data, stderr_data)
    else:
      sys.exit('Error in input: ' + str(self.make_command))

if __name__ == '__main__':

  #l = 'execve("/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc", ["/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc", "-ccbin=clang++", "-restrict", "-gencode=arch=compute_70,code=sm_70", "-O3", "--expt-extended-lambda", "-Xcompiler=-fPIC", "-Wno-deprecated-gpu-targets", "-shared", "-dlink", "CMakeFiles/kripke.exe.dir/src/kripke.cpp.o", "-o", "CMakeFiles/kripke.exe.dir/cmake_device_link.o", "-L/usr/tce/packages/cuda/cuda-9.2.148/nvidia/targets/ppc64le-linux/lib/stubs", "-L/usr/tce/packages/cuda/cuda-9.2.148/nvidia/targets/ppc64le-linux/lib", "lib/libchai.a", "lib/libRAJA.a", "/usr/tce/packages/cuda/cuda-9.2.148/lib64/libcudart_static.a", "-lpthread", "-ldl", "lib/libkripke.a", "lib/libumpire.a", "-lcudadevrt", "-lcudart_static", "-lrt"], 0x7fffffffb8b8 /* 129 vars */) = 0\n'
  #strace = CommandsTracing(['make', '-j'])
  #ret = strace.isTopCommand(l)
  #print(l)
  #print('ret:', ret)
  #exit()

  #strace.analyzeTraces()
  #strace.traced_commands.append(('', l))
  #strace.writeToFile()
  #exit()

  cmd = sys.argv[1:]
  strace = CommandsTracing(cmd)
  #strace.startTracing()
  strace.analyzeTraces()
  strace.writeToFile()
