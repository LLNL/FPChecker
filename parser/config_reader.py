
import os
from configparser import ConfigParser
from collections import defaultdict

class Config:
  def __init__(self, conf_file: str):
    self.config_file = conf_file
    self.omitted_lines = defaultdict(list)
    if os.path.isfile(conf_file):
      self._read_config()

  def _read_config(self):
    cfg = ConfigParser()
    cfg.read(self.config_file)

    ## Get omitted lines
    lines = cfg.get('omit', 'omit_lines')
    lines = lines.replace('\n', '')
    lines = ''.join(lines.split())
    lines = lines.split(',')
    for l in lines:
      file_name = l.split(':')[0]
      lines_range = l.split(':')[1]
      x = int(lines_range.split('-')[0])
      y = int(lines_range.split('-')[1])
      self.omitted_lines[file_name].append((x,y))
      #print('In file', file_name, 'omit', lines_range)

  def is_line_omitted(self, file_name: str, line: int) -> bool:
    if file_name in self.omitted_lines:
      for r in self.omitted_lines[file_name]:
        if line >= r[0] and line <= r[1]:
          return True
    return False

if __name__ == '__main__':
  #c = Config('test.ini')
  c = Config('fpchecker.ini')
  print('compute.cu, 30', c.is_line_omitted('compute.cu', 30))
  print('compute.cu, 31', c.is_line_omitted('compute.cu', 31))
  print('compute.cu, 55', c.is_line_omitted('compute.cu', 55))
  print('newfile.cu, 55', c.is_line_omitted('newfile.cu', 55))
