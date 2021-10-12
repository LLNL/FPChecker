import glob
import json
import sys

def loadReport(fileName):
  f = open(fileName,'r')
  data = json.load(f)
  f.close()
  return data

# ----- Regular reports ------
def findReportFile(path):
  reports = glob.glob(path+'/fpc_*.json')
  return reports[0]

def numberReportFiles(path):
  reports = glob.glob(path+'/fpc_*.json')
  return len(reports)

# ------ Histogram reports -------
def findHistogramFile(path):
  reports = glob.glob(path+'/histogram_*.json')
  return reports[0]

if __name__ == '__main__':
  fileName = sys.argv[1]
  loadReport(fileName)
