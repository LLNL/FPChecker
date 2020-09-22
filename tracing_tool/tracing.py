import subprocess
import sys
import re
from colors import prGreen,prCyan,prRed

STRACE = '/usr/workspace/wsa/laguna/strace/strace/strace'

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
  'xlc++_r'
])

SUPPORTED_TOOLS = set([
  'ar'
])

# Examples of top commands
# [pid 83362] execve("/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc", 
# [pid 63885] execve("/bin/sh", ["/bin/sh", "-c", "cd /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/build/src/util && /usr/tcetmp/bin/c++     -o CMakeFiles/util.dir/util.cpp.o -c /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/src/util/util.cpp"]

# Saves Compilation commands
class CommandsTracing:

  #open("/usr/tcetmp/packages/spack/opt/spack/linux-redhat7-ppc64le/gcc-4.8.5/gcc-4.9.3-3clrxj5wz2i54h
  #[pid  8690] execve("/usr/tcetmp/bin/c++", ["/usr/tcetmp/bin/c++", "CMakeFiles/main.dir/src/main.cpp.o", "-o", "main"]
  pidPattern = re.compile("^\[pid\s+[0-9]+\] ")

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
      #child = line.split('Process')[1].split()[0]
      #child = child.replace(' ','')
      
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
      if line.endswith('/'+compiler):
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
          #print('is is sh')
          cmd = line.split('["')[1].split(']')[0]
          cmd = cmd.replace(',','')
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

  def writeToFile(self):
    fileNameRaw = './raw_traces.txt'
    prGreen('Saving raw traces in '+fileNameRaw)
    fd = open(fileNameRaw, 'w')
    for line in self.traced_commands:
      fd.write(line+'\n')
    fd.close()

    fileNameExec = './executable_traces.txt'
    prGreen('Saving executable traces in '+fileNameExec)
    fd = open(fileNameExec, 'w')
    for line in self.traced_commands:
      line = line.split(', [')[1:]
      line = ' '.join(line).split(']')[0]
      line = line.replace(',','')
      line = line.replace('"', '')
      line = line.replace('\\', '')
      if '/sh -c' in line:
        line = ' '.join(line.split()[2:]) # remove /bin/sh -c

      if '-E ' in line: # Remove commands that only run the preprocessor with -E
        continue

      fd.write(line+'\n')
    fd.close()

    #print('\nself.childTree\n')
    #for p in self.childTree:
    #  print("\t", p, '==>', self.childTree[p])

    #print('\nself.parentTree:')
    #for c in self.parentTree:
    #  print('\t', c, '-->', self.parentTree[c])

  def replayTraces(self, fileName):
    fd = open(fileName, 'r')
    for line in fd:
      self.saveCompilingCommands(line)
    fd.close()

  def startTracing(self):
    fd = open('./all_traces.txt', 'w')
    #trace_command = [STRACE, '-e', 'quiet=attach,exit,path-resolution,personality,thread-execve', '-e', 'trace=desc,process', '-f', '-s', '9999'] + self.make_command
    trace_command = [STRACE, '-e', 'quiet=attach,exit,path-resolution,personality,thread-execve', '-f', '-s', '9999'] + self.make_command
    process = subprocess.Popen(trace_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Poll process for new output until finished
    while True:
      #nextline = process.stdout.readline()
      nextline = process.stderr.readline()

      if process.poll() is not None:
        break
      
      l = nextline.decode('utf-8')
      self.saveCompilingCommands(l)
      fd.write(l)

    (stdout_data, stderr_data) = process.communicate()
    exitCode = process.returncode

    if (exitCode == 0):
      return (stdout_data, stderr_data)
    else:
      prCyan('Error exit code: '+str(exitCode))
      print('Error in:', command)
      exit(-1)

    fd.close()


if __name__ == '__main__':
  cmd = sys.argv[1:]
  strace = CommandsTracing(cmd)
  strace.startTracing()
  strace.writeToFile()

  #strace = CommandsTracing([])
  #strace.replayTraces(sys.argv[1])
  #strace.writeToFile()

  #strace = CommandsTracing([])
  #line = "[pid 26114] vfork(strace: Process 26115 attached"
  #strace.buildChildTree(line)

  #l1 = '[pid 83362] execve("/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc",...' 
  #l2 = '[pid 63885] execve("/bin/sh", ["/bin/sh", "-c", "cd /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/build/src/util && /usr/tcetmp/bin/c++     -o CMakeFiles/util.dir/util.cpp.o -c /usr/workspace/wsa/laguna/fpchecker/FPChecker/tests/tracing_tool/dynamic/test_cmake_simple/src/util/util.cpp"]'
  #strace = CommandsTracing([])
  #print(strace.isTopCommand(l1))
  #print(strace.isTopCommand(l2))

  #l1 = '[pid 129570] <... clone resumed>child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x200000044f60) = 129601'
  #strace = CommandsTracing([])
  #strace.buildChildTree(l1)
