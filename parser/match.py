
import sys
from tokenizer import Tokenizer, Token, SymbolToken, KeywordToken, WhiteSpaceToken, IdentifierToken

#--------------------------------------------------------------------#
# Match                                                              #
#--------------------------------------------------------------------#

class Match:
  def __init__(self):
    pass

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

  def _match_anything_until(self, buff, untilStr):
    for i in range(len(buff)):
      if str(buff[i])==untilStr:
        return i+1
    return False

  ## Match anything we see the last '}' 
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

  ## Returns a line number range that defines a device function:
  ##
  ##   __attribute__((device)) ANY ( ANY ) ANY { ANY }
  ##
  def match_device_function(self, buff):
    linesThatMatched = []
    startIndexes = [] # index of __attribute__ tokens
    for i in range(len(buff)):
      if self._match_keyword(buff[i], '__attribute__'):
        startIndexes.append(i)

    ## Iterate starting from potential blocks
    for i in startIndexes:
      if i+1+10 > len(buff): # we need at least 10 token to match
        continue

      m1 = self._match_device_decl(buff[i:])
      if m1:
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
                linesThatMatched.append((startLine, endLine, startIndex, endIndex))

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
        print('PRE:', tokensRange[i+1+left])
        print('POST:', tokensRange[i+m1-1])
        tokenIndexes.append((i+1+left, i+m1-1))
    return tokenIndexes

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
    startLine, endLine, startIndex, endIndex = l # unpack lines and indexes
    tokenIndexes = m.match_assigment(allTokens[startIndex:endIndex])

