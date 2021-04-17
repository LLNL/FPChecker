
import os
from colors import prRed, prGreen

def verbose() -> bool:
  if 'FPC_VERBOSE' in os.environ:
    return True
  return False

## Saves message in log file
def logMessage(msg: str):
  with open('.fpc_log.txt', 'a') as fd:
    fd.write(msg + '\n')

