#!/usr/bin/env python3

# Description: This script creates an html report of all the events.
#              It assumes that event (json) files are created by each 
#              MPI process indepdently. 

import os
import argparse
import sys
import json
from collections import defaultdict
import shutil 
from line_highlighting import createHTMLCode
from colors import prGreen, prCyan, prRed

# -------------------------------------------------------- #
# Insertion points
# -------------------------------------------------------- #
P_INFINITY_POS = '<!-- INFINITY_POS -->'
P_INFINITY_NEG = '<!-- INFINITY_NEG -->'
P_NAN = '<!-- NAN -->'
P_DIV_ZERO = '<!-- DIV_ZERO -->'
P_CANCELLATION = '<!-- CANCELLATION -->'
P_COMPARISON = '<!-- COMPARISON -->'
P_UNDERFLOW = '<!-- UNDERFLOW -->'
P_LATENT_INFINITY_POS = '<!-- LATENT_INFINITY_POS -->'
P_LATENT_INFINITY_NEG = '<!-- LATENT_INFINITY_NEG -->'
P_LATENT_UNDERFLOW = '<!-- LATENT_UNDERFLOW -->'
P_CODE_PATHS = '<!-- CODE_PATHS -->'
P_FILES_AFFECTED = '<!-- FILES_AFFECTED -->'
P_LINES_AFFECTED = '<!-- LINES_AFFECTED -->'
P_REPORT_TITLE = '<!-- REPORT_TITLE -->' 

# -------------------------------------------------------- #
# PATHS
# -------------------------------------------------------- #

TRACES_DIR = '.fpc_logs'
REPORTS_DIR = './fpc-report'
ROOT_REPORT_NAME = 'index.html'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_REPORT_TEMPLATE_DIR = THIS_DIR+'/../cpu_checking/report_templates'
ROOT_REPORT_TEMPLATE = ROOT_REPORT_TEMPLATE_DIR+'/index.html' 
EVENT_REPORT_TEMPLATE = ROOT_REPORT_TEMPLATE_DIR+'/event_report_template.html'
SOURCE_REPORT_TEMPLATE = ROOT_REPORT_TEMPLATE_DIR+'/source_report_template.html' 

# -------------------------------------------------------- #
# Globals
# -------------------------------------------------------- #
report_title = ""
events = defaultdict(lambda: defaultdict(list) )

def getEventFilePaths(p):
  fileList = []
  for root, dirs, files in os.walk(p):
    for file in files:
      fileName = os.path.split(file)[1]
      if fileName.startswith('fpc_') and fileName.endswith(".json"):
        f = str(os.path.join(root, file))
        fileList.append(f)
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
      positive_infinity    = data[i]['infinity_pos']
      negative_infinity    = data[i]['infinity_neg']
      nan             = data[i]['nan']
      division_by_zero   = data[i]['division_zero']
      cancellation    = data[i]['cancellation']
      comparison      = data[i]['comparison']
      underflow       = data[i]['underflow']
      latent_positive_infinity = data[i]['latent_infinity_pos']
      latent_negative_infinity = data[i]['latent_infinity_neg']
      latent_underflow    = data[i]['latent_underflow']

      if positive_infinity != int(0): events['positive_infinity'][fileName].append((line,positive_infinity))
      if negative_infinity != int(0): events['negative_infinity'][fileName].append((line,negative_infinity))
      if nan != int(0): events['nan'][fileName].append((line,nan))
      if division_by_zero != int(0): events['division_by_zero'][fileName].append((line,division_by_zero))
      if cancellation != int(0): events['cancellation'][fileName].append((line,cancellation))
      if comparison != int(0): events['comparison'][fileName].append((line,comparison))
      if underflow != int(0): events['underflow'][fileName].append((line,underflow))
      if latent_positive_infinity != int(0): events['latent_positive_infinity'][fileName].append((line,latent_positive_infinity))
      if latent_negative_infinity != int(0): events['latent_negative_infinity'][fileName].append((line,latent_negative_infinity))
      if latent_underflow != int(0): events['latent_underflow'][fileName].append((line,latent_underflow))

def getEvents(event_type):
  n = 0
  for f in events[event_type]:
    for l in events[event_type][f]:
      n += int(l[1])
  return n

def getCodePaths():
  files = set([])
  for e in events:
    for f in events[e]:
      files.add(f)
  return os.path.commonpath(list(files))

def getFilesAffected():
  files = set([])
  for e in events:
    for f in events[e]:
      files.add(f)
  return len(files)

def getLinesAffected():
  lines = set([])
  for e in events:
    for f in events[e]:
      for t in events[e][f]:
        lines.add((f, t[0]))
  return len(lines)

def createRootReport():
  if os.path.exists(REPORTS_DIR):
    prRed('Overwriting report dir...')
    shutil.rmtree(REPORTS_DIR)
  os.mkdir(REPORTS_DIR)

  # Load template
  fd = open(ROOT_REPORT_TEMPLATE, 'r')
  templateLines = fd.readlines()
  fd.close()

  # Copy style and other files
  shutil.copy2(ROOT_REPORT_TEMPLATE_DIR+'/sitestyle.css', REPORTS_DIR+'/sitestyle.css')
  if not os.path.exists(REPORTS_DIR+'/icons_3'):
    shutil.copytree(ROOT_REPORT_TEMPLATE_DIR+'/icons_3', REPORTS_DIR+'/icons_3')

  report_full_name = REPORTS_DIR+'/'+ROOT_REPORT_NAME 
  fd = open(report_full_name, 'w')
  for i in range(len(templateLines)):
    if P_INFINITY_POS in templateLines[i]:
      e = getEvents('positive_infinity')
      if e != 0:
        fd.write('<a href="./positive_infinity/positive_infinity.html">'+str(e)+'</a>\n')
        createEventReport('positive_infinity')
      else: fd.write(str(e)+'\n')

    elif P_INFINITY_NEG in templateLines[i]:
      e = getEvents('negative_infinity')
      if e != 0:
        fd.write('<a href="./negative_infinity/negative_infinity.html">'+str(e)+'</a>\n')
        createEventReport('negative_infinity')
      else: fd.write(str(e)+'\n')

    elif P_NAN in templateLines[i]:
      e = getEvents('nan')
      if e != 0: 
        fd.write('<a href="./nan/nan.html">'+str(e)+'</a>\n')
        createEventReport('nan')
      else: fd.write(str(e)+'\n')

    elif P_DIV_ZERO in templateLines[i]:
      e = getEvents('division_by_zero')
      if e != 0:
        fd.write('<a href="./division_by_zero/division_by_zero.html">'+str(e)+'</a>\n')
        createEventReport('division_by_zero')
      else: fd.write(str(e)+'\n')

    elif P_CANCELLATION in templateLines[i]:
      e = getEvents('cancellation')
      if e != 0: 
        fd.write('<a href="./cancellation/cancellation.html">'+str(e)+'</a>\n')
        createEventReport('cancellation')
      else: fd.write(str(e)+'\n')

    elif P_COMPARISON in templateLines[i]:
      e = getEvents('comparison')
      if e != 0:
        fd.write('<a href="./comparison/comparison.html">'+str(e)+'</a>\n')
        createEventReport('comparison')
      else: fd.write(str(e)+'\n')

    elif P_UNDERFLOW in templateLines[i]:
      e = getEvents('underflow')
      if e != 0: 
        fd.write('<a href="./underflow/underflow.html">'+str(e)+'</a>\n')
        createEventReport('underflow')
      else: fd.write(str(e)+'\n')

    elif P_LATENT_INFINITY_POS in templateLines[i]:
      e = getEvents('latent_positive_infinity')
      if e != 0:
        fd.write('<a href="./latent_positive_infinity/latent_positive_infinity.html">'+str(e)+'</a>\n')
        createEventReport('latent_positive_infinity')
      else: fd.write(str(e)+'\n')

    elif P_LATENT_INFINITY_NEG in templateLines[i]:
      e = getEvents('latent_negative_infinity')
      if e != 0:
        fd.write('<a href="./latent_negative_infinity/latent_negative_infinity.html">'+str(e)+'</a>\n')
        createEventReport('latent_negative_infinity')
      else: fd.write(str(e)+'\n')

    elif P_LATENT_UNDERFLOW in templateLines[i]:
      e = getEvents('latent_underflow')
      if e != 0:
        fd.write('<a href="./latent_underflow/latent_underflow.html">'+str(e)+'</a>\n')
        createEventReport('latent_underflow')
      else: fd.write(str(e)+'\n')

    elif P_CODE_PATHS in templateLines[i]:
      fd.write(getCodePaths()+'\n')

    elif P_LINES_AFFECTED in templateLines[i]:
      fd.write(str(getLinesAffected())+'\n')

    elif P_FILES_AFFECTED in templateLines[i]:
      fd.write(str(getFilesAffected())+'\n')
    
    elif P_REPORT_TITLE in templateLines[i]:
      fd.write(report_title+'\n')
     
    else:
        fd.write(templateLines[i])

  fd.close()

  prGreen('Report created: ' + report_full_name)

def createEventReport(event_name):
  report_name = (' '.join(event_name.split('_'))).title()
  
  # Load template
  fd = open(EVENT_REPORT_TEMPLATE, 'r')
  templateLines = fd.readlines()
  fd.close()

  if not os.path.exists(REPORTS_DIR+'/'+event_name):
    os.mkdir(REPORTS_DIR+'/'+event_name)

  fd = open(REPORTS_DIR+'/'+event_name+'/'+event_name+'.html', 'w')
  source_id = 0
  for i in range(len(templateLines)):
    if '<!-- PAGE_NAME -->' in templateLines[i]:
      fd.write(report_name)
    elif '<!-- REPORT_NAME -->' in templateLines[i]:
      fd.write(report_name+' Report')
    elif '<!-- FILE_ENTRIES -->' in templateLines[i]:
      for file in events[event_name]:
        lines = set([])
        for t in events[event_name][file]:
          lines.add((file, t[0]))
        source_id += 1
        fd.write('<tr><td class="files_class">'+file+'</td>\n')
        fd.write('<td class="files_class"><a href="./source_'+str(source_id)+'.html">')
        fd.write(str(len(lines))+'</a></td></tr>')
        createCodeReport(event_name, file, source_id)
    elif P_REPORT_TITLE in templateLines[i]:
      fd.write(report_title+'\n')
    else:
      fd.write(templateLines[i])
  fd.close()

def createCodeReport(event_name, file_full_path, id):
  report_name = (' '.join(event_name.split('_'))).title()
  
  # Load template
  fd = open(SOURCE_REPORT_TEMPLATE, 'r')
  templateLines = fd.readlines()
  fd.close()
  
  fd = open(REPORTS_DIR+'/'+event_name+'/source_'+str(id)+'.html', 'w')
  for i in range(len(templateLines)):
    if '<!-- EVENT_REPORT_HTML -->' in templateLines[i]:
      fd.write('<a href="./'+event_name+'.html">')
    elif '<!-- EVENT_REPORT_NAME -->' in templateLines[i]:
      fd.write(report_name+'\n')
    elif '<!-- FILE_FULL_PATH -->' in templateLines[i]:
      fd.write(file_full_path+'\n')
    elif '<!-- FILE_NAME -->' in templateLines[i]:
      fd.write(os.path.split(file_full_path)[1]+'\n')
    elif '<!-- CODE_LINE -->' in templateLines[i]:
      highligth_set = set([])
      for t in events[event_name][file_full_path]:
        highligth_set.add(int(t[0]))
      htmlCode = createHTMLCode(file_full_path, highligth_set)
      for l in htmlCode:
        fd.write(l+'\n')
    elif P_REPORT_TITLE in templateLines[i]:
      fd.write(report_title+'\n')
    else:
      fd.write(templateLines[i])
  fd.close()

def removeReportDir():
  if os.path.exists(REPORTS_DIR):
    prRed('Removing report dir...')
    shutil.rmtree(REPORTS_DIR)
  else:
    prGreen('There is no report directory to remove.')

# Sample query file
#[
#  {
#  "file": "test.cpp",
#  "line": "any",
#  "infinity_pos": 0,
#  "infinity_neg": 1,
#  "nan": 0,
#  "division_zero": 0,
#  "cancellation": 1,
#  "comparison": 0,
#  "underflow": 0,
#  "latent_infinity_pos": 0,
#  "latent_infinity_neg": 0,
#  "latent_underflow": 0
#  }
#]
def executeQuery(fileName):
  prGreen('Loading: ' + fileName)
  fd = open(fileName, 'r')
  data = json.load(fd)
  fd.close()

  # Walk on the dir to find trace files
  current_path = './'
  for root, dirs, files in os.walk(current_path):
    for file in files:
      fname = os.path.split(file)[1]
      if fname.startswith('fpc_') and fname.endswith(".json"):
        f = str(os.path.join(root, file))
        with open(f, 'r') as trace_file:
          trace_data = json.load(trace_file)
          for i in trace_data:
            if i["file"].endswith(data[0]["file"]):
                if (data[0]['infinity_pos'] <= i['infinity_pos'] and
                    data[0]['infinity_neg'] <= i['infinity_neg'] and
                    data[0]['nan'] <= i['nan'] and
                    data[0]['division_zero'] <= i['division_zero'] and
                    data[0]['cancellation'] <= i['cancellation'] and
                    data[0]['comparison'] <= i['comparison'] and
                    data[0]['underflow'] <= i['underflow'] and
                    data[0]['latent_infinity_pos'] <= i['latent_infinity_pos'] and
                    data[0]['latent_infinity_neg'] <= i['latent_infinity_neg'] and
                    data[0]['latent_underflow'] <= i['latent_underflow']
                    ): 
                  print('Trace:', f)

def removeTraces():
  p = './'
  for root, dirs, files in os.walk(p):
    for d in dirs:
      if d.endswith(TRACES_DIR):
        full_path = str(os.path.join(root, d))
        prGreen('Removing: ' + full_path)
        try:
          shutil.rmtree(full_path)
        except Exception as e:
          prRed(e)
 
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='FPChecker report generator')
  parser.add_argument('-r', '--remove', action='store_true', help='Remove report dir.')
  parser.add_argument('-c', '--clean', action='store_true', help='Remove traces. A report cannot be generated without traces.')
  parser.add_argument('-t', '--title', nargs=1, type=str, help='Title of report.')
  parser.add_argument('-q', '--query', nargs=1, type=str, action='store', help='Query file.')
  parser.add_argument('dir', nargs='?', default=os.getcwd())
  args = parser.parse_args()

  if (args.query):
    fileName = args.query[0]
    executeQuery(fileName)
    exit()
 
  if (args.remove or args.clean):
    if (args.remove):
      removeReportDir()
    if (args.clean):
      removeTraces()
    exit()
    
  if (args.title):
    report_title = args.title[0]
    
  reports_path = args.dir  
  prCyan('Generating FPChecker report...')
  fileList = getEventFilePaths(reports_path)
  print('Trace files found:', len(fileList))
  loadEvents(fileList)
  createRootReport()
