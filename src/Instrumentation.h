/*
 * Instrumentation.h
 *
 *  Created on: Apr 11, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_INSTRUMENTATION_H_
#define SRC_INSTRUMENTATION_H_

#include "CommonTypes.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace CUDAAnalysis {

class FPInstrumentation
{
private:

  Module *mod;

  Function *fp32_check_add_function;
  Function *fp32_check_sub_function;
  Function *fp32_check_mul_function;
  Function *fp32_check_div_function;

  Function *fp64_check_add_function;
  Function *fp64_check_sub_function;
  Function *fp64_check_mul_function;
  Function *fp64_check_div_function;

  /* Host */
  Function *print_at_main;

  IRBuilder<> createBuilderAfter(Instruction *inst);
  IRBuilder<> createBuilderBefore(Instruction *inst);
  void setFakeDebugLocation(Function *f, Instruction *inst);
  Instruction* firstInstrution();

public:
  FPInstrumentation(Module *M);
  void instrumentFunction(Function *f);
  void instrumentMainFunction(Function *f);
  void generateCodeForInterruption();

  /* Helper functions */
  //static bool isUnwantedFunction(Function *f);
  static bool isFPOperation(const Instruction *inst);
  static bool isDoubleFPOperation(const Instruction *inst);
  static bool isSingleFPOperation(const Instruction *inst);
  //static bool isMainFunction(Function *f);
};

}


#endif /* SRC_INSTRUMENTATION_H_ */
