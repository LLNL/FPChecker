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
#include <string>

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

  Function *_fpc_interrupt_;
  Function *_fpc_warning_;
  Function *_fpc_print_errors_;

  // device globals
  bool errors_dont_abort = false;

  // maximum number for a code line
  int maxNumLocations = 0;

  /* Host */
  Function *print_at_main = nullptr;

  IRBuilder<> createBuilderAfter(Instruction *inst);
  IRBuilder<> createBuilderBefore(Instruction *inst);
  void setFakeDebugLocation(Function *f, Instruction *inst);
  Instruction* firstInstrution();

  GlobalVariable* generateIntArrayGlobalVariable(ArrayType *arrType);
  void createReadFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName);
  void createWriteFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName);


public:
  FPInstrumentation(Module *M);
  void instrumentFunction(Function *f);
  void instrumentMainFunction(Function *f);
  void generateCodeForInterruption();
  void instrumentErrorArray();
  void instrumentEndOfKernel(Function *f);
  //InstSet finalInstrutions(Function *f);

  /* Helper functions */
  //static bool isUnwantedFunction(Function *f);
  static bool isFPOperation(const Instruction *inst);
  static bool isDoubleFPOperation(const Instruction *inst);
  static bool isSingleFPOperation(const Instruction *inst);
  //static bool isMainFunction(Function *f);
  bool errorsDontAbortMode();
};

}


#endif /* SRC_INSTRUMENTATION_H_ */
