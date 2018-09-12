/*
 * Utility.cpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"

#include "Utility.h"

#include <set>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

using namespace llvm;

namespace CUDAAnalysis {

void printMessage(const char *s)
{
	//errs() << "[ERR_INJ] " << s << "\n";
}

std::string inst2str(const Instruction *i)
{
	std::string s;
	raw_string_ostream rso(s);
	i->print(rso);
	return "{" + rso.str() + "}";
}

std::string getInstructionInformation(const Instruction *i)
{
  std::stringstream lineStr("");

  if (DILocation *Loc = i->getDebugLoc()) // only true if dbg info exists
  {
    unsigned Line = Loc->getLine();
    StringRef File = Loc->getFilename();
    StringRef Dir = Loc->getDirectory();
    lineStr << Dir.str() << "/" << File.str() << ":"
        << NumberToString<unsigned>(Line);
  }
  else
  {
    lineStr << "NONE";
  }

  return lineStr.str().c_str();
}

int getLineOfCode(const Instruction *i)
{
	int ret = -1;
  if (DILocation *loc = i->getDebugLoc()) // only true if dbg info exists
  {
    ret = (int)loc->getLine();
  }
  return ret;
}

std::string getFileNameFromInstruction(const Instruction *i)
{
  std::stringstream lineStr("");

  if (DILocation *Loc = i->getDebugLoc()) // only true if dbg info exists
  {
    StringRef File = Loc->getFilename();
    StringRef Dir = Loc->getDirectory();
    lineStr << Dir.str() << "/" << File.str();
  }
  else
  {
    lineStr << "Unknown";
  }

  return lineStr.str().c_str();
}

std::string getFileNameFromModule(const Module *mod)
{
	return mod->getModuleIdentifier();
}

bool mayModifyMemory(const Instruction *i)
{
	return (
			isa<StoreInst>(i) ||
			isa<AtomicCmpXchgInst>(i) ||
			isa<AtomicRMWInst>(i)
			);
}

void tokenize(const std::string &str,
    std::vector<std::string> &tokens,
    const std::string &delimiters)
{
  // Skip delimiters at beginning.
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // Find first "non-delimiter".
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);

  while (std::string::npos != pos || std::string::npos != lastPos)
  {
    // Found a token, add it to the vector.
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    // Skip delimiters.  Note the "not_of"
    lastPos = str.find_first_not_of(delimiters, pos);
    // Find next "non-delimiter"
    pos = str.find_first_of(delimiters, lastPos);
  }
}

bool isFunctionUnwanted(const std::string &str)
{
  if (str.find("_LOG_FLOATING_POINT_OP_") != std::string::npos) return true;
  if (str.find("dumpShadowValues") != std::string::npos) return true;
  if (str.find("_PRINT_TABLE_") != std::string::npos) return true;
  return false;
}

void stop()
{
	char input;
	scanf("%c", & input);
}

}
