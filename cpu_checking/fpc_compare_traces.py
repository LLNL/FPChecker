#!/usr/bin/env python3

import sys
from copy import deepcopy
import fpc_create_report

def checkLineExists(eventsDict, event_name, fileName, lineNumber):
  if event_name in eventsDict:
    if fileName in eventsDict[event_name]:
      for t in eventsDict[event_name][fileName]:
        line = t[0]
        if line == lineNumber:
          return True
  return False

if __name__ == '__main__':
  dir1 = sys.argv[1]
  dir2 = sys.argv[2]
  print('Comparing:')
  print('\t', dir1)
  print('\t', dir2)

  fileList = fpc_create_report.getEventFilePaths(dir1)
  print('Trace files found (dir 1):', len(fileList))
  fpc_create_report.loadEvents(fileList)
  events_dir1 = deepcopy(fpc_create_report.events)

  fpc_create_report.events.clear()

  fileList = fpc_create_report.getEventFilePaths(dir2)
  print('Trace files found (dir 2):', len(fileList))
  fpc_create_report.loadEvents(fileList)
  events_dir2 = deepcopy(fpc_create_report.events)

  #print(events_dir1)
  #print(events_dir2)

  for event_name in events_dir2.keys():
    print('=====', event_name, "=====")
    cases = set([])
    for fileName in events_dir2[event_name]:
      for t in events_dir2[event_name][fileName]:
        line = t[0]
        # Check if this event occurred in dir1
        if not checkLineExists(events_dir1, event_name, fileName, line):
          cases.add(fileName+':'+str(line))

    for c in cases:
      print('\t'+c)
