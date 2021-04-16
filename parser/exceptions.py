

class FPCheckerException(Exception):
  pass

class MatchException(FPCheckerException):
  pass

class CommandException(FPCheckerException):
  pass

class CompileException(FPCheckerException):
  pass

class EmptyFileException(FPCheckerException):
  pass
