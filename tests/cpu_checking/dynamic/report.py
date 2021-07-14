import glob
import json
import sys

def loadReport(fileName):
  f = open(fileName,'r')
  data = json.load(f)
  f.close()
  return data

def findReportFile(path):
  reports = glob.glob(path+'/fpc_*.json')
  return reports[0]

if __name__ == '__main__':
  fileName = sys.argv[1]
  loadReport(fileName)
