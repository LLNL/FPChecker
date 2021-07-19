#!/usr/bin/env python3

# Description: This script creates an html report of all the events.
#              It assumes that event (json) files are created by each 
#              MPI process indepdently. The script merges the json 
#              files into one json file.

import os
import sys
import json
from collections import defaultdict

# Sample json file entry:
#  {
#  "file": "/path/file_1.cpp",
#  "line": 1,
#  "infinity_pos": 1,
#  "infinity_neg": 1,
#  "nan": 1,
#  "division_zero": 1,
#  "cancellation": 1,
#  "comparison": 1,
#  "underflow": 1,
#  "latent_infinity_pos": 1,
#  "latent_infinity_neg": 1,
#  "latent_underflow": 1
#  },

REPORTS_DIR = './fpc-report'
ROOT_REPORT_NAME = 'index.html'
events = defaultdict(lambda: defaultdict(list) )

def getEventFilePaths(p):
  fileList = []
  for root, dirs, files in os.walk(p):
    for file in files:
      fileName = os.path.split(file)[1]
      if fileName.startswith('fpc_') and fileName.endswith(".json"):
        f = str(os.path.join(root, file))
        fileList.append(f)
        #print(f)
  return fileList

def loadReport(fileName):
  f = open(fileName,'r')
  data = json.load(f)
  f.close()
  return data

def loadEvents(files):
  for f in files:
    data = loadReport(f)
    for i in range(len(data)):
      fileName        = data[i]['file']
      line            = data[i]['line']
      infinity_pos    = data[i]['infinity_pos']
      infinity_neg    = data[i]['infinity_neg']
      nan             = data[i]['nan']
      division_zero   = data[i]['division_zero']
      cancellation    = data[i]['cancellation']
      comparison      = data[i]['comparison']
      underflow       = data[i]['underflow']
      latent_infinity_pos = data[i]['latent_infinity_pos']
      latent_infinity_neg = data[i]['latent_infinity_neg']
      latent_underflow    = data[i]['latent_underflow']

      events['infinity_pos'][fileName].append((line,infinity_pos))
      events['infinity_neg'][fileName].append((line,infinity_neg))
      events['nan'][fileName].append((line,nan))
      events['division_zero'][fileName].append((line,division_zero))
      events['cancellation'][fileName].append((line,cancellation))
      events['comparison'][fileName].append((line,comparison))
      events['underflow'][fileName].append((line,underflow))
      events['latent_infinity_pos'][fileName].append((line,latent_infinity_pos))
      events['latent_infinity_neg'][fileName].append((line,latent_infinity_neg))
      events['latent_underflow'][fileName].append((line,latent_underflow))

def createRootReport():
  if not os.path.exists(REPORTS_DIR):
    os.mkdir(REPORTS_DIR)

  fd = open(REPORTS_DIR+'/'+ROOT_REPORT_NAME, 'w')
  fd.write('<table>\n')
  for e in events:
    fd.write('\t<tr>\n')
    fd.write('\t\t<td>')
    fd.write(e)
    total = 0
    for f in events[e]:
      for t in events[e][f]:
        _, num = t
        total += int(num)
    fd.write('</td>\n')
    fd.write('\t\t<td>')
    fd.write(str(total))
    fd.write('</td>\n')
    fd.write('\t</tr>\n')
  
  fd.write('</table>\n')
  fd.close()

if __name__ == '__main__':
  reports_path = sys.argv[1]
  fileList = getEventFilePaths(reports_path)
  loadEvents(fileList)
  createRootReport()
