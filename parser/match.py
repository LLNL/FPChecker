
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
      n = 10 # number of whispaces
      for i in range(n):
        if not self._match_white_space(buff[h+i]):
          break
      d = self._match_device_decl(buff[h+i:])
      if d:
        return h+i+1+d
    return False

  ## Matches __attribute__((device)) __attribute__((host)) 
  def _match_device_host_decl(self, buff):
    d = self._match_device_decl(buff)
    if d:
      n = 10 # number of whispaces
      for i in range(n):
        if not self._match_white_space(buff[d+i]):
          break
      h = self._match_host_decl(buff[d+i:])
      if h:
        return d+i+1+h
    return False

  ## Matches __attribute__((host)) __attribute__((device)) 
#  def _match_host_device_decl(self, buff):
#    if len(buff) < 12:
#      return False
#    if (self._match_keyword(buff[0], '__attribute__') and
#        self._match_symbol(buff[1], '(') and
#        self._match_symbol(buff[2], '(') and
#        self._match_identifier(buff[3], 'host') and
#        self._match_symbol(buff[4], ')') and
#        self._match_symbol(buff[5], ')') and
#        self._match_white_space(buff[6]) and
#        self._match_keyword(buff[7], '__attribute__') and
#        self._match_symbol(buff[8], '(') and
#        self._match_symbol(buff[9], '(') and
#        self._match_identifier(buff[10], 'device') and
#        self._match_symbol(buff[11], ')') and
#        self._match_symbol(buff[12], ')')
#         ):
#      return 12
#    return False

  ## Matches __attribute__((device)) __attribute__((host)) 
#  def _match_device_host_decl(self, buff):
#    if len(buff) < 12:
#      return False
#    if (self._match_keyword(buff[0], '__attribute__') and
#        self._match_symbol(buff[1], '(') and
#        self._match_symbol(buff[2], '(') and
#        self._match_identifier(buff[3], 'device') and
#        self._match_symbol(buff[4], ')') and
#        self._match_symbol(buff[5], ')') and
#        self._match_white_space(buff[6]) and
#        self._match_keyword(buff[7], '__attribute__') and
#        self._match_symbol(buff[8], '(') and
#        self._match_symbol(buff[9], '(') and
#        self._match_identifier(buff[10], 'host') and
#        self._match_symbol(buff[11], ')') and
#        self._match_symbol(buff[12], ')')
#         ):
#      return 12
#    return False

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

  def _match_anything_until_except(self, buff, untilStr, exceptChars):
    for i in range(len(buff)):
      if str(buff[i]) in exceptChars:
        return False
      if str(buff[i])==untilStr:
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
        if isinstance(d, int): m1 = d
        elif isinstance(d_h, int): m1 = d_h
        elif isinstance(h_d, int): m1 = h_d

        m2 = self._match_anything_until(buff[i+m1:], '(')
        if m2:
          m3 = self._match_anything_until(buff[i+m1+m2:], ')')
          if m3:
            m4 = self._match_anything_until(buff[i+m1+m2+m3:], '{')
            if m4:
              m5 = self._match_anything_until_balanced_bracket(buff[i+m1+m2+m3+m4:])
              if m5:
                startLine = buff[i].lineNumber()
                endLine = buff[i+m1+m2+m3+m4+m5].lineNumber()
                startIndex = i
                endIndex = i+m1+m2+m3+m4+m5
                if not self._matched_block( (startLine, endLine) ):
                  if verbose():
                    print('Not seen block:', (startLine, endLine), '\ncache:', self.code_range_cache)
                  linesThatMatched.append((startLine, endLine, startIndex, endIndex, func_type))

    return linesThatMatched

  ## Matches an assigment in a given range of lines:
  ## ... = x + y ...;
  ##
  ## Returns list of tuples (i, j), where with the indexes
  ## of the begin anf end of the assigment RHS (right hand side)
  ## for each assigment in the range.
  def match_assigment(self, tokensRange):
    startIndexes = [] # indexes with assigment operator
    for i in range(len(tokensRange)):
      if (self._match_symbol(tokensRange[i], '=') or
          self._match_symbol(tokensRange[i], '+=') or
          self._match_symbol(tokensRange[i], '-=') or
          self._match_symbol(tokensRange[i], '*=') or
          self._match_symbol(tokensRange[i], '/=')
        ):
        startIndexes.append(i)

    tokenIndexes = []
    unallowedChars = set(['{', '}'])
    for i in startIndexes:
      ## Match until ; except if we see a body {}
      m1 = self._match_anything_until_except(tokensRange[i:], ';', unallowedChars)
      if m1:
        left = self._nextNonEmpty(tokensRange[i+1:])
        if verbose(): print('PRE:', tokensRange[i+1+left])
        if verbose(): print('POST:', tokensRange[i+m1-1])
        tokenIndexes.append((i+1+left, i+m1-1))
    return tokenIndexes

  # Print friendly a bugger of tokens
  def printTokens(self, buff):
    for i in range(len(buff)):
      if verbose(): print('['+str(i)+']:', str(buff[i]))
    
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

