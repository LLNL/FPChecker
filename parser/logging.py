

def logMessage(msg: str):
  with open('.fpc_log.txt', 'a') as fd:
    fd.write(msg + '\n')


