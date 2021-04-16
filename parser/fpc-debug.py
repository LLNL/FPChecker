import os

def getLogFiles() -> list:
  fileList = []
  for root, dirs, files in os.walk("./"):
    for file in files:
      if file.endswith(".fpc_log.txt"):
        f = str(os.path.join(root, file))
        fileList.append(f)
        #print(f)
  return fileList

def removeFiles(fileList: list):
  for f in fileList:
    print('Removing:', f)
    os.remove(f)

def report(fileList: list):
  inst = 0
  failed = 0
  for f in fileList:
    with open(f, 'r') as fd:
      for line in fd:
        if line.startswith('Instrumented'):
          inst += 1
        elif line.startswith('Failed:'):
          failed += 1

  print('*** FPChcecker Report ***')
  print('Instrumented files:', inst)
  print('Failed:', failed)

if __name__ == '__main__':
  fileList = getLogFiles()
  #removeFiles(fileList)
  report(fileList)
