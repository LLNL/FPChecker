
# Padding lines: lines printed but not highlighted
# Dotted lines: only show dots
def calc_lines_to_highligh(file_len:int, highligth_set:set):
  padding_set = set([])
  dots_set = set([])
  for l in highligth_set:
    before = l-1
    after = l+1
    dots_before = before-1
    dots_after = after+1
    if before >= 1:
      padding_set.add(before)
    if after <= file_len:
      padding_set.add(after)
    if dots_before >= 1:
      dots_set.add(dots_before)
    if dots_after <= file_len:
      dots_set.add(dots_after)

  #d = defaultdict(char)
  d = {}
  for i in range(file_len):
    line = i+1
    if line in dots_set:
      d[line] = 'D'
    if line in padding_set:
      d[line] = 'P'
    if line in highligth_set:
      d[line] = 'H'

  return d
  
def replaceCodeChars(line):
  l = line.replace('&', '&amp;')
  l = l.replace('<', '&lt;')
  l = l.replace('>', '&gt;')
  return l
  
def createHTMLCode(file_full_path:str, highligth_set:set):
  fd = open(file_full_path, 'r')
  all_lines = fd.readlines()
  fd.close()
  
  ret = []
  d = calc_lines_to_highligh(len(all_lines), highligth_set)
  for k in d:
    line = '<tr><td class="code_line_class">'+str(k)+'</td>'
    if d[k] == 'D':
      line = line + '<td><code> ... </code></td></tr>'
    elif d[k] == 'P':
      line = line + '<td><code>'+all_lines[k-1][:-1]+'</code></td></tr>'
    elif d[k] == 'H':
      line = line + '<td><span class="highlightme"><code>'+all_lines[k-1][:-1]+'</code></span></td></tr>'
    ret.append(line)
  
  return ret
  
if __name__ == '__main__':
  highligth_set = set([4,5])
  file_full_path = '/Users/lagunaperalt1/projects/fpchecker/FPChecker/cpu_checking/test.c'
  lines = createHTMLCode(file_full_path, highligth_set)
  for l in lines:
    print(l)
