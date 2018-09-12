/*
 * Instrumentation.h
 *
 *  Created on: Jul 20, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef CODE_SRC_INSTRUMENTATION_H_
#define CODE_SRC_INSTRUMENTATION_H_

#include "CommonTypes.h"

#include <map>
#include <string>

using namespace llvm;

namespace CUDAAnalysis {


class Instrumentation
{
private:
  // Functions handler
	Value *_LOG_FLOATING_POINT_OP_;
  //Value *_LOG_DOUBLE_ADDITION_OP_;
  //Value *_LOG_FLOAT_ADDITION_OP_;

  Module *mod;

  StringMap stringTable;
  int stringHash(const char *str );

public:
  Instrumentation(Module *M);

  /// Insert instrumentation function to this instruction
  /// Returns true if succeeded
  //bool instrumentFPAddition(Instruction *inst, const char *str);
  bool instrumentFPOperation(Instruction *inst, const char *str);
  //bool instrumentFPOperation(Instruction *inst, const char *str);
  //bool instrumentIntOperation(Instruction *inst, const char *str);
  //bool instrumentCall(Instruction *inst, const char *str);

  /// Insert instrumentation function before MPI_Finalize
  void instrumentFinalize(Instruction *inst);

  void changeFPInstructions(StringSet instSet, Function *f);

  void printTable() const;
};

typedef struct InstructionData_ {
	Instruction *fp64Inst;
	Instruction *fp32Inst;
} InstructionData;


class FPInstrumentation
{
private:

  Module *mod;
  //DominatorTree *domTree;
 ValToValMap castMap;

  static std::string removeDebugInfo(const std::string &instruction);
  InstSet getInstTree(const StringSet &instSet, Function *f);
  InsToInsMap createFP32Instructions(const InstSet &s);
  void connectInstructions(const InsToInsMap &m);
  bool isInstructionInTree(const Instruction *fp64Inst, const InsToInsMap &m, Instruction **fp32Inst) const;
  void changeInstructionOperand(Instruction *target, Value *op_old, Value *op_new) const;
  void changeInstructionOperandWithIndex(Instruction *target, int i, Value *op_new) const;
  void removeFP64Instructions(InstSet &fp64Tree);
  InstHashTable getMapOfInstructionsInFunction(Function *f) const;
  void reorderPHINodes(Function *f);
  Value* createCastAtBeginning(Value *inInst, Function *f);
  Value* createCast(Value *inInst);


public:
  //FPInstrumentation(Module *M) : mod(M), domTree(nullptr) {};
  FPInstrumentation(Module *M) : mod(M) {};

  void changeFPInstructions(const StringSet &instSet, double ratio, Function *f);
};

}

#endif /* CODE_SRC_INSTRUMENTATION_H_ */
