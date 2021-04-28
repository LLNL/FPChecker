
import re

prog_1 = """
__device__ void comp(double *x) {
  double y=0.0, z;
  x[0] = y*z; /* this is
 a multi line 
comment*/
}
"""

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

if __name__ == '__main__':
  new_prog = comment_remover(prog_1)
  print(new_prog)
