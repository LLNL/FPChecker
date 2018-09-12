/*
 * CommonTypes.h
 *
 *  Created on: Jul 15, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef CODE_SRC_COMMONTYPES_H_
#define CODE_SRC_COMMONTYPES_H_

#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"

#include <set>
#include <list>
#include <map>
#include <unordered_map>

using namespace llvm;

namespace CUDAAnalysis {

///-- Set of instructions --
//typedef std::set<const Instruction *> InstSet;
typedef std::set<Instruction *> InstSet;

///-- List of instructions --
typedef std::list<const Instruction *> InstList;

///-- Map of Instructions --> Set of instructions->Set
typedef std::map<const Instruction *, InstSet> InstMap;

///-- Map of Instructions --> Set of instructions->Instruction
typedef std::map<Instruction *, Instruction *> InsToInsMap;

///-- Map of Values
typedef std::map<Value *, Value *> ValToValMap;

///-- Set of Store instructions
typedef std::set<const StoreInst *> StoresSet;

///-- Map of Instructions --> String
typedef std::map<const Instruction*, std::string> InstToStringMap;

///-- Map of Instructions --> bool
typedef std::map<const Instruction*, bool> InstToBoolMap;

///-- Set of strings --
typedef std::set<std::string> StringSet;

///-- Pair of (real, Set of strings)
typedef std::pair<double, StringSet> StringSetPair;

///-- Pair of (Function *, StringSetPair)
typedef std::pair<Function *, StringSetPair> StringSetFunctionPair;

typedef std::unordered_map<std::string, Instruction *> InstHashTable;

}

#endif /* CODE_SRC_COMMONTYPES_H_ */
