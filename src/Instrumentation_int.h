/*
 * Instrumentation_int.h
 *
 *  Created on: Mar 13, 2019
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#ifndef SRC_INSTRUMENTATION_INT_H_
#define SRC_INSTRUMENTATION_INT_H_

#include "CommonTypes.h"
#include "llvm/IR/IRBuilder.h"
#include <string>

using namespace llvm;

namespace CUDAAnalysis {

class IntegerInstrumentation
{
private:

  Module *mod;

  Function *int32_check_add_function;
  Function *int32_check_sub_function;
  Function *int32_check_mul_function;
  Function *int32_check_div_function;

  //Function *fp64_check_add_function;
  //Function *fp64_check_sub_function;
  //Function *fp64_check_mul_function;
  //Function *fp64_check_div_function;


  Function *_fpc_init_htable_;
  Function *_fpc_print_locations_;


  // maximum number for a code line
  int maxNumLocations = 0;

  IRBuilder<> createBuilderAfter(Instruction *inst);
  IRBuilder<> createBuilderBefore(Instruction *inst);
  void setFakeDebugLocation(Function *f, Instruction *inst);
  Instruction* firstInstrution();

  //GlobalVariable* generateIntArrayGlobalVariable(ArrayType *arrType);
  //void createReadFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName);
  //void createWriteFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName);


public:
  IntegerInstrumentation(Module *M);
  void instrumentFunction(Function *f);
  void instrumentMainFunction(Function *f);
  //void generateCodeForInterruption();
  //void instrumentErrorArray();
  //void instrumentEndOfKernel(Function *f);
  //InstSet finalInstrutions(Function *f);

  /* Helper functions */
  //static bool isUnwantedFunction(Function *f);
  static bool isIntOperation(const Instruction *inst);

  static bool is64BitIntOperation(const Instruction *inst);
  static bool is32BitIntOperation(const Instruction *inst);
  //static bool isMainFunction(Function *f);
};

}




#endif /* SRC_INSTRUMENTATION_INT_H_ */
