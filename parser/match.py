
import sys
import enum
from tokenizer import Tokenizer, Token, SymbolToken, KeywordToken, WhiteSpaceToken, IdentifierToken
from fpc_logging import verbose, logMessage

#--------------------------------------------------------------------#
# Types                                                              #
#--------------------------------------------------------------------#
class FunctionType(enum.Enum):
  host = 0
  device = 1
  host_device = 2
  device_host = 3

#--------------------------------------------------------------------#
# Match                                                              #
#--------------------------------------------------------------------#

class Match:
  def __init__(self):
    # Ranges of code blocks that have been matched: (b, e)
    # b: block begining, e: block end
    self.code_range_cache = []

  def _matched_block(self, blockRange: tuple) -> bool:
    if len(self.code_range_cache) == 0:
      self.code_range_cache.append(blockRange)
      return False
    x, y = blockRange
    last_x, last_y = self.code_range_cache[-1:][0]
    if x >= last_x and y <= last_y:
      return True
    else:
      # Block has not been seen
      self.code_range_cache.append(blockRange)
    return False

  def _match_keyword(self, token, content):
    if isinstance(token, KeywordToken):
      if str(token)==content:
        return True
    return False

  def _match_symbol(self, token, content):
    if isinstance(token, SymbolToken):
      if str(token)==content:
        return True
    return False

  def _match_identifier(self, token, content):
    if isinstance(token, IdentifierToken):
      if str(token)==content:
        return True
    return False

  def _match_white_space(self, token):
    if isinstance(token, WhiteSpaceToken):
      return True
    return False

  ## Matches __attribute__((device))
  def _match_device_decl(self, buff):
    if len(buff) < 6:
      return False
    if (self._match_keyword(buff[0], '__attribute__') and
        self._match_symbol(buff[1], '(') and
        self._match_symbol(buff[2], '(') and
        self._match_identifier(buff[3], 'device') and
        self._match_symbol(buff[4], ')') and
        self._match_symbol(buff[5], ')')
        ):
      return 6
    return False

  ## Matches __attribute__((host))
  def _match_host_decl(self, buff):
    if len(buff) < 6:
      return False
    if (self._match_keyword(buff[0], '__attribute__') and
        self._match_symbol(buff[1], '(') and
        self._match_symbol(buff[2], '(') and
        self._match_identifier(buff[3], 'host') and
        self._match_symbol(buff[4], ')') and
        self._match_symbol(buff[5], ')')
        ):
      return 6
    return False

  ## Matches __attribute__((host)) __attribute__((device)) 
  def _match_host_device_decl(self, buff):
    h = self._match_host_decl(buff)
    if h:
      n = 20 # number of whispaces
      for i in range(n):
        if not self._match_white_space(buff[h+i]):
          break
      d = self._match_device_decl(buff[h+i:])
      if d:
        return h+i+d
    return False

  ## Matches __attribute__((device)) __attribute__((host)) 
  def _match_device_host_decl(self, buff):
    d = self._match_device_decl(buff)
    if d:
      n = 20 # number of whispaces
      for i in range(n):
        if not self._match_white_space(buff[d+i]):
          break
      h = self._match_host_decl(buff[d+i:])
      if h:
        return d+i+h
    return False

  def _match_anything_until(self, buff, untilStr):
    for i in range(len(buff)):
      if str(buff[i])==untilStr:
        return i+1
    return False

  ## Match anything until we see the last '}' 
  def _match_anything_until_balanced_bracket(self, buff):
    openBrackets = 0
    for i in range(len(buff)):
      if str(buff[i])=='{':
        openBrackets += 1
      if str(buff[i])=='}':
        openBrackets -= 1
      if openBrackets == -1:
        return i+1
    return False

  ## Match anything until we see a particular token, e.g., ';'
  ## If see an unwanted char, it doesn't match and return False
  def _match_anything_until_except(self, buff, untilStr, exceptChars):
    for i in range(len(buff)):
      if str(buff[i]) in exceptChars:
        return False
      if str(buff[i])==untilStr:
        return i+1
    return False

  ## Match anything until we see a particular token, e.g., ';',
  ## or until we see an inbalanaced parethesis ')'.
  ## If we see an assigment operator (=) we return False
  ## If we see a curly bracket ({) we return False
  def _match_anything_until_or_imbalanced_parenthesis(self, buff, untilStr):
    open_parenthesis = 0      # (
    open_square_brackets = 0  # [
    open_less_than = 0        # <
    equal_sign = 0
    found_match = False
    for i in range(len(buff)):
      if str(buff[i])=='<':   open_less_than += 1
      elif str(buff[i])=='>': open_less_than -= 1
      elif str(buff[i])=='[': open_square_brackets += 1
      elif str(buff[i])==']': open_square_brackets -= 1
      elif str(buff[i])=='(': open_parenthesis += 1
      elif str(buff[i])==')':
        open_parenthesis -= 1
        if open_parenthesis == -1: # found imbalanced parenthesis
          #return i+1
          found_match = True
          break
      elif str(buff[i])=='=':
        equal_sign += 1
        if equal_sign > 1:
          return False
      elif str(buff[i])==untilStr:
        if open_parenthesis == 0:
          found_match = True
          break
          #return i+1
      elif str(buff[i])=='{' or str(buff[i])=='}': return False
      elif str(buff[i])==',':
        if (open_parenthesis==0 and 
            open_square_brackets==0 and
            open_less_than==0):
          found_match = True
          break
          #return i+1

    if found_match:
      # Check there is at least one arithmetic operator
      for k in range(0, i+1):
        t = buff[k]
        if str(t)=='+' or str(t)=='-' or str(t)=='*' or str(t)=='/':
          return i+1

    return False

  ## Get next non-white-space token
  def _nextNonEmpty(self, tokensList):
    for i in range(len(tokensList)):
      if not isinstance(tokensList[i], WhiteSpaceToken):
        return i
    return None

  ## Match any of these three device annotations:
  ##  (a) __device__
  ##  (b) __device__ __host__
  ##  (c) __host__ __device__
  def _match_any_device_annotation(self, buff):
    d = False   # device
    dh = False  # device host
    hd = False  # host device
    func_type = FunctionType.host
    dh = self._match_device_host_decl(buff)
    if not dh:
      dh = self._match_host_device_decl(buff)
      if not dh:
        d = self._match_device_decl(buff)
        if d:
          func_type = FunctionType.device
      else:
        func_type = FunctionType.host_device
    else:
      func_type = FunctionType.device_host
    return d, dh, hd, func_type

  ## Returns a line number range that defines a device function:
  ##
  ##   __attribute__((device)) ANY ( ANY ) ANY { ANY }
  ##   __attribute__((device)) __attribute__((host)) ANY ( ANY ) ANY { ANY }
  ##   __attribute__((device)) __attribute__((host)) ANY ( ANY ) ANY { ANY }
  ##
  def match_device_function(self, buff):
    linesThatMatched = []
    startIndexes = [] # index of __attribute__ tokens
    for i in range(len(buff)):
      if self._match_keyword(buff[i], '__attribute__'):
        startIndexes.append(i)

    ## Iterate starting from potential function definitions
    for i in startIndexes:
      if i+1+10 > len(buff): # we need at least 10 token to match
        continue

      #m1 = self._match_device_decl(buff[i:])
      d, d_h, h_d, func_type = self._match_any_device_annotation(buff[i:])
      if d or d_h or h_d:
        # Get the number of tokens fromn the annotation that matched
        if d: m1 = d
        elif d_h: m1 = d_h
        elif h_d: m1 = h_d
        else: return [] # we couldn't match any device function
        #print('m1', m1, 'func_type', func_type)

        m2 = self._match_anything_until(buff[i+m1:], '(')
        if m2:
          m3 = self._match_anything_until(buff[i+m1+m2:], ')')
          if m3:
            m4 = self._match_anything_until(buff[i+m1+m2+m3:], '{')
            if m4:
              m5 = self._match_anything_until_balanced_bracket(buff[i+m1+m2+m3+m4:])
              if m5:
                startIndex = i
                endIndex = i+m1+m2+m3+m4+m5 - 1 # Important to subtract 1 (because indexes start with zero)
                startLine = buff[i].lineNumber()
                endLine = buff[endIndex].lineNumber()
                if not self._matched_block( (startLine, endLine) ):
                  if verbose():
                    print('Not seen block:', (startLine, endLine), '\ncache:', self.code_range_cache)
                  linesThatMatched.append((startLine, endLine, startIndex, endIndex, func_type))

    return linesThatMatched

  def _find_indexes_with_assignmets(self, tokensRange):
    startIndexes = [] # indexes with assigment operator
    for i in range(len(tokensRange)):
      if (self._match_symbol(tokensRange[i], '+=') or
          self._match_symbol(tokensRange[i], '-=') or
          self._match_symbol(tokensRange[i], '*=') or
          self._match_symbol(tokensRange[i], '/=')
        ):
        startIndexes.append(i)
      elif (self._match_symbol(tokensRange[i], '=')):
        if not (self._match_symbol(tokensRange[i-1], '<') or
                self._match_symbol(tokensRange[i-1], '>') or
                self._match_symbol(tokensRange[i-1], '=')):
          startIndexes.append(i)
    return startIndexes

  ## Matches an assigment in a given range of lines:
  ## ... = x + y ...;
  ##
  ## Returns list of tuples (i, j), where with the indexes
  ## of the begin anf end of the assigment RHS (right hand side)
  ## for each assigment in the range.
  def match_assigment(self, tokensRange):
    startIndexes = self._find_indexes_with_assignmets(tokensRange)
    ret = []
    for i in startIndexes:
      ## Match until ;
      m1 = self._match_anything_until_or_imbalanced_parenthesis(tokensRange[i:], ';')
      if m1:
        left = self._nextNonEmpty(tokensRange[i+1:])
        if verbose(): print('PRE:', tokensRange[i+1+left])
        if verbose(): print('POST:', tokensRange[i+m1-1])
        ret.append((i+1+left, i+m1-1))
    return ret

  # Print friendly a bugger of tokens
  def printTokens(self, buff):
    for i in range(len(buff)):
      if verbose(): print('['+str(i)+']:', str(buff[i]))
    
#--------------------------------------------------------------------#
# Helper                                                             #
#--------------------------------------------------------------------#

def printTokenBuffer(buff):
  print('*** Buffer ***')
  for i in range(len(buff)):
    print(i, ':', buff[i].__class__.__name__, buff[i])

#--------------------------------------------------------------------#
# Main                                                               #
#--------------------------------------------------------------------#

if __name__ == '__main__':
  fileName = sys.argv[1]
  t = Tokenizer(fileName)
  allTokens = []
  for token in t.tokenize():
    allTokens.append(token)
    print('token', type(token), ':', str(token), 'line:', token.lineNumber())

  m = Match()
  funcLines = m.match_device_function(allTokens)
  print('Lines:', funcLines)

  for l in funcLines:
    startLine, endLine, startIndex, endIndex, _ = l # unpack lines and indexes
    tokenIndexes = m.match_assigment(allTokens[startIndex:endIndex])

