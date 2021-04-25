
import sys
import os
import re
import tempfile
from exceptions import MatchException, EmptyFileException, TokenException
from fpc_logging import verbose, logMessage

# Lookahead tokens: 1
CPP_SYMBOL_L1 = set([
  '&',
  '|',
  '~',
  '!',
  '^',
  '{',
  '}',
  '[',
  ']',
  ':',
  ';',
  '.',
  '+',
  '-',
  '*',
  '/',
  '?',
  '=',
  '>',
  '<',
  '%',
  ')',
  '(',
  ','
])

# Lookahead tokens: 2
CPP_SYMBOL_L2 = set([
  '&&',
  '&=',
  '!=',
  '||',
  '|=',
  '^=',
  '::',
  '>>',
  '<<',
  '+=',
  '-=',
  '*=',
  '/=',
  '>>',
  '++',
  '--',
  '**',
  '->'
])

# Lookahead tokens: 3
CPP_SYMBOL_L3 = set([
  '>>=',
  '<<=',
  '<<<',
  '>>>'
])

CPP_KEYWORD = set([
  'alignas',
  'alignof',
  'and',
  'and_eq',
  'asm',
  'atomic_cancel',
  'atomic_commit',
  'atomic_noexcept',
  'auto',
  'bitand',
  'bitor',
  'bool',
  'break',
  'case',
  'catch',
  'char',
  'char8_t',
  'char16_t',
  'char32_t',
  'class',
  'compl',
  'concept',
  'const',
  'consteval',
  'constexpr',
  'constinit',
  'const_cast',
  'continue',
  'co_await',
  'co_return',
  'co_yield',
  'decltype',
  'default',
  'delete',
  'do',
  'double',
  'dynamic_cast',
  'else',
  'enum',
  'explicit',
  'export',
  'extern',
  'false',
  'float',
  'for',
  'friend',
  'goto',
  'if',
  'inline',
  'int',
  'long',
  'mutable',
  'namespace',
  'new',
  'noexcept',
  'not',
  'not_eq',
  'nullptr',
  'operator',
  'or',
  'or_eq',
  'private',
  'protected',
  'public',
  'reflexpr',
  'register',
  'reinterpret_cast',
  'requires',
  'return',
  'short',
  'signed',
  'sizeof',
  'static',
  'static_assert',
  'static_cast',
  'struct',
  'switch',
  'synchronized',
  'template',
  'this',
  'thread_local',
  'throw',
  'true',
  'try',
  'typedef',
  'typeid',
  'typename',
  'union',
  'unsigned',
  'using',
  'virtual',
  'void',
  'volatile',
  'wchar_t',
  'while',
  'xor',
  'xor_eq',
  'final',
  'override',
  'transaction_safe',
  'transaction_safe_dynamic',
  '__attribute__'
])

#--------------------------------------------------------------------#
# Token classes                                                      #
#--------------------------------------------------------------------#

identifierPattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
numberPattern     = re.compile(r'[0-9.]')

class Token:
  def __init__(self, t: list, l: int):
    self.token = ''.join(t)
    self.line = l # line number

  def __str__(self):
    return self.token

  def lineNumber(self):
    return self.line

  def areEqual(self, t):
    return self.token == t.token

class SymbolToken(Token):
  def __init__(self, t: str, l: int):
    if (t not in CPP_SYMBOL_L1
      and t not in CPP_SYMBOL_L2
      and t not in CPP_SYMBOL_L3):
      raise TokenException('Error: unknown symbol token')
    super().__init__(t, l)

class KeywordToken(Token):
  def __init__(self, t: str, l: int):
    if t not in CPP_KEYWORD:
      raise TokenException('Error: unknown keyword token')
    super().__init__(t, l)

class WhiteSpaceToken(Token):
  def __init__(self, t: str, l: int):
    if (t[0]!=' ' and
      t[0]!='\t' and 
      t[0]!='\n' and
      t[0]!='\r'):
      raise TokenException('Error: not a white space token')
    super().__init__(t, l)

class IdentifierToken(Token):
  def __init__(self, t: str, l: int):
    idMatch = identifierPattern.match(t)
    numMatch = numberPattern.match(t)
    if idMatch == None and numMatch == None:
      raise TokenException('Error: not an identifier token: ' + t)
    super().__init__(t, l)

class StringToken(Token):
  def __init__(self, t: str, l: int):
    if t[0] != '"' or t[-1:][0] != '"':
      print('--->', t)
      raise TokenException('Error: not a string token')
    super().__init__(t, l)

class CharToken(Token):
  def __init__(self, t: str, l: int):
    if t[0] != "'" or t[-1:][0] != "'":
      raise TokenException('Error: not a char token')
    super().__init__(t, l)

#--------------------------------------------------------------------#
# Tokenizer                                                          #
#--------------------------------------------------------------------#

class Tokenizer:
  def __init__(self, fileName: str):
    self.fileName = os.path.abspath(fileName)
    self.buff = []
    self.current_line = 1

  def tokenize(self):
    ## Create temp file and remove pre-processor lines (start with #)
    tmpFd, tmpFname = tempfile.mkstemp(suffix='.txt', text=True)
    if verbose(): print('Temp file (tokenizer):', tmpFname)
    with open(tmpFname, 'w') as f:
      with open(self.fileName, 'r') as src:
        for l in src:
          if l.startswith('#'):
            f.write('\n')
          else:
            f.write(l)

    ## Iterate on new file one character at a time
    with open(tmpFname, 'r') as f:
      while True:
        c = f.read(1)
        if not c:
          ## Match the final 2 tokens
          self.buff.append('\n')
          self.buff.append('\n')
          while True:
            token = self.match(self.buff)
            if not token or len(self.buff)==2: break
            else: yield token

          if verbose(): print("\nEnd of file")
          break
       
        self.buff.append(c)
        token = self.match(self.buff)
        if token:
          yield token
        if token != None:
          continue

    os.close(tmpFd)
    if 'FPC_LEAVE_TEMP_FILES' not in os.environ:
      os.remove(tmpFname)

  def match(self, buff: str):
    if len(buff)==0:
      raise EmptyFileException('Buffer len=0 in tokenizer - this file is empty')

    ### First let's try to match white spaces
    if Tokenizer.is_white_space(buff):
      self.consume(1)
      # Update line number
      if buff[0] == '\n':
        self.current_line += 1
      return WhiteSpaceToken(buff[0], self.current_line)

    if len(buff) < 3:
      return None
    
    ### Try to match symbols
    l3 = self.match_symbol_l3(buff)
    if l3: return l3
    l2 = self.match_symbol_l2(buff)
    if l2: return l2
    l1 = self.match_symbol_l1(buff)
    if l1: return l1
   
    ## If we can't match white spaces and/or symbols
    ## we match strings and chars
    if buff[0] == '"':
      string = self.match_string(buff)
      return string 
    if buff[0] == "'":
      char = self.match_char(buff)
      return char
 
    ### If at this point we can't match white spaces or symbols,
    ### we try to match keywords.
    ### We look for the pattern ...[white_space] or ...[symbol]
    ### where ... is a keyword.
    if Tokenizer.endsWithDelimiter(buff):
      k = self.match_keyword(buff)
      if k: return k
    else:
      return None

    ### If we couldn't match white spaces, symbols, or keywords,
    ### the token must be an identifier.
    ident = self.match_identifier(buff)
    if ident: return ident

    return None

  def match_identifier(self, buff: str):
    keyword = ''
    for c in buff:
      keyword += c
      delim = Tokenizer.endsWithDelimiter(keyword)
      if delim:
        keyword = keyword[:-delim]
        self.consume(len(keyword))
        return IdentifierToken(keyword, self.current_line)
    return None 

  ## Returns either a srtring token or None
#  def match_string(self, buff: str):
#    if buff[0] == '"' and buff[-1:][0] == '"':
#      self.consume(len(buff))
#      return StringToken(buff, self.current_line)
#    return None
  def match_string(self, buff: str):
    if buff[0] == '"':
      for i in range(1, len(buff)):
        if buff[i] == '"':
          self.consume(i+1)
          return StringToken(buff[0:i+1], self.current_line)
    return None

  ## Returns either a srtring token or None
  def match_char(self, buff: str):
    if buff[0] == "'" and buff[-1:][0] == "'":
      self.consume(len(buff))
      return CharToken(buff, self.current_line)
    return None

  def match_keyword(self, buff: str):
    keyword = ''
    for c in buff:
      keyword += c
      delim = Tokenizer.endsWithDelimiter(keyword)
      if delim:
        keyword = keyword[:-delim]
        if keyword in CPP_KEYWORD:
          self.consume(len(keyword))
          return KeywordToken(keyword, self.current_line)
    return None 

  def match_symbol_l3(self, buff: str):
    sym = ''.join(buff[:3])
    if sym in CPP_SYMBOL_L3:
      self.consume(3)
      return SymbolToken(sym, self.current_line)
    return None

  def match_symbol_l2(self, buff: str):
    sym = ''.join(buff[:2])
    if sym in CPP_SYMBOL_L2:
      self.consume(2)
      return SymbolToken(sym, self.current_line)
    return None

  def match_symbol_l1(self, buff: str):
    if buff[0] in CPP_SYMBOL_L1:
      self.consume(1)
      return SymbolToken(buff[0], self.current_line)
    return None

  def consume(self, n: int):
    if n > 0 and n <= len(self.buff):
      self.buff = self.buff[n:]

  @staticmethod
  def is_white_space(char: str):
    if char[0]==' ' or char[0]=='\t' or char[0]=='\n' or char=='\r':
      return True
    return False

  @staticmethod
  def ends_with_symbol(buff: str):
    sym = ''.join(buff[-3:])
    if sym in CPP_SYMBOL_L3:
      return 3
    sym = ''.join(buff[-2:])
    if sym in CPP_SYMBOL_L2:
      return 2
    sym = ''.join(buff[-1:])
    if sym in CPP_SYMBOL_L1:
      return 1
    return False

  @staticmethod
  def endsWithDelimiter(buff: str):
    if Tokenizer.is_white_space(buff[-1:]):
      return 1
    sym = Tokenizer.ends_with_symbol(buff)
    if sym: return sym 
    return False

#--------------------------------------------------------------------#
# Main                                                               #
#--------------------------------------------------------------------#

if __name__ == '__main__':
  fileName = sys.argv[1]
  l = Tokenizer(fileName)
  try:
    for token in l.tokenize():
      sys.stdout.write('\n'+str(type(token))+':')
      sys.stdout.write(str(token))
  except Exception as e:
    print(e)
