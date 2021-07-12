
#include "Utility.h"
#include "Instrumentation_cpu.h"
#include "CodeMatching.h"
#include "Logging.h"
//#include "CommonTypes.h"
//#include "ProgressBar.h"

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/LegacyPassManager.h"

#include <string>
#include <iostream>
#include <fstream>
#include <set>

using namespace llvm;

namespace CPUAnalysis
{

class CPUKernelAnalysis : public ModulePass
{
public:
  static char ID;

  CPUKernelAnalysis() : ModulePass(ID) {}

	virtual bool runOnModule(Module &M)
	{
		Module *m = &M;
		CPUFPInstrumentation *fpInstrumentation = new CPUFPInstrumentation(m);

#ifdef FPC_DEBUG
		std::string out = "Running Module pass on: " + m->getName().str();
		CUDAAnalysis::Logging::info(out.c_str());
#endif

		for (auto f = M.begin(), e = M.end(); f != e; ++f) {
			// Discard function declarations
			if (f->isDeclaration())
				continue;

			Function *F = &(*f);

      if (CUDAAnalysis::CodeMatching::isUnwantedFunction(F))
          continue;

#ifdef FPC_DEBUG
      std::string fname = "Instrumenting function: " + F->getName().str();
      CUDAAnalysis::Logging::info(fname.c_str());
#endif
      fpInstrumentation->instrumentFunction(F);

      if (CUDAAnalysis::CodeMatching::isMainFunction(F)) {
#ifdef FPC_DEBUG
        CUDAAnalysis::Logging::info("main() found");
#endif
        fpInstrumentation->instrumentMainFunction(F);
      }
		}

		delete fpInstrumentation;
		return false;
	}

};

char CPUKernelAnalysis::ID = 0;

static RegisterPass<CPUKernelAnalysis> X(
		"cpukernels",
		"CPUKernelAnalysis Pass",
		false,
		false);

static void registerPass(const PassManagerBuilder &, legacy::PassManagerBase &PM)
{
	PM.add(new CPUKernelAnalysis());
}

static RegisterStandardPasses
    RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,registerPass);
static RegisterStandardPasses
    RegisterMyPass2(PassManagerBuilder::EP_OptimizerLast,registerPass);
}




