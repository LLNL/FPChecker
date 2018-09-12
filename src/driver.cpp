
#include "Utility.h"
#include "Instrumentation.h"
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

  bool isFunctionAKernel(Function *f)
  {
  	Module *m = f->getParent();
  	NamedMDNode *NMD = m->getNamedMetadata("nvvm.annotations");
		if (!NMD)
			return false;

		for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
		{
			const MDNode *md = NMD->getOperand(i);

			GlobalValue *entity =
					mdconst::dyn_extract_or_null<GlobalValue>(md->getOperand(0));
			// entity may be null due to DCE
			if (!entity)
				continue;
			if (entity != f)
				continue;

		  assert(md && "Invalid mdnode for annotation");
		  assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
			for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
			    // property
			    const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
			    assert(prop && "Annotation property not a string");

			    ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(md->getOperand(i + 1));
			    unsigned v = Val->getZExtValue();
			    std::string keyname = prop->getString().str();
			    if (keyname.find("kernel") != std::string::npos)
			    	return (v == 1);
			}
		}

		// Last resort: check the calling convention
		return (f->getCallingConv() == CallingConv::PTX_Kernel || f->getCallingConv() == CallingConv::PTX_Device);
  }

	virtual bool runOnModule(Module &M)
	{
		Module *m = &M;
		FPInstrumentation *fpInstrumentation = new FPInstrumentation(m);
		outs() << "Running Module pass on module: " << m->getName().str() << "\n";

		for (auto f = M.begin(), e = M.end(); f != e; ++f)
		{
			// Discard function declarations
			if (f->isDeclaration())
				continue;

			Function *F = &(*f);
			if (isFunctionAKernel(F)) {
				if (FPInstrumentation::isUnwantedFunction(F))
						continue;

				outs() << "Instrumenting func: " << f->getName().str() << "\n";
				fpInstrumentation->instrumentFunction(F);

			}
		}

		fpInstrumentation->generateCodeForInterruption();

		delete fpInstrumentation;
		return false;
	}

};

//MachinePassRegistry RegisterMyPasses::Registry;

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
    RegisterMyPass(PassManagerBuilder::EP_OptimizerLast,registerPass);
}




