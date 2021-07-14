

class FPCheckerException(Exception):
  pass

class TokenException(FPCheckerException):
  pass

class MatchException(FPCheckerException):
  pass

class CommandException(FPCheckerException):
  pass

class CompileException(FPCheckerException):
  pass

class EmptyFileException(FPCheckerException):
  pass

class DepreprocessorException(FPCheckerException):
  pass
