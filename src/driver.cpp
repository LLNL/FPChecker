
#include "Utility.h"
#include "Instrumentation.h"
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
//#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
//#include "llvm/lib/Target/NVPTX/NVPTXUtilities.h"


#include <string>
#include <iostream>
#include <fstream>
#include <set>

using namespace llvm;
//using namespace std;

namespace CUDAAnalysis
{

class CUDAKernelAnalysis : public ModulePass
{
public:
  static char ID;

  CUDAKernelAnalysis() : ModulePass(ID) {}

	virtual bool runOnModule(Module &M)
	{
		Module *m = &M;
		FPInstrumentation *fpInstrumentation = new FPInstrumentation(m);

#ifdef FPC_DEBUG
		std::string out = "Running Module pass on: " << m->getName().str();
		Logging::info(out.c_str());
#endif

		for (auto f = M.begin(), e = M.end(); f != e; ++f)
		{
			// Discard function declarations
			if (f->isDeclaration())
				continue;

			Function *F = &(*f);

			if (CodeMatching::isDeviceCode(m))
			{
				if (CodeMatching::isUnwantedFunction(F))
						continue;

#ifdef FPC_DEBUG
				std::string out = "Instrumenting function: " + f->getName().str();
				Logging::info(out.c_str());
#endif
				fpInstrumentation->instrumentFunction(F);
			}
		}
		
		if (CodeMatching::isDeviceCode(m))
			fpInstrumentation->generateCodeForInterruption();

		delete fpInstrumentation;
		return false;
	}

};

char CUDAKernelAnalysis::ID = 0;

static RegisterPass<CUDAKernelAnalysis> X(
		"cudakernels",
		"CUDAKernelAnalysis Pass",
		false,
		false);

static void registerPass(const PassManagerBuilder &, legacy::PassManagerBase &PM)
{
	PM.add(new CUDAKernelAnalysis());
}

static RegisterStandardPasses
    RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,registerPass);
static RegisterStandardPasses
    RegisterMyPass2(PassManagerBuilder::EP_OptimizerLast,registerPass);
}




