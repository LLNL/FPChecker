
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
//#include "llvm/lib/Target/NVPTX/NVPTXUtilities.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/ManagedStatic.h"

#include <string>
#include <iostream>
#include <fstream>
#include <set>

using namespace llvm;
//using namespace std;

namespace CUDAAnalysis
{

namespace {
typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;
} // anonymous namespace
static ManagedStatic<per_module_annot_t> annotationCache;
static sys::Mutex Lock;

class CUDAKernelAnalysis : public ModulePass
{
public:
  static char ID;

  CUDAKernelAnalysis() : ModulePass(ID) {}

  /*bool isFunctionAKernel(Function *f)
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
  }*/

  static void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval) {
    MutexGuard Guard(Lock);
    assert(md && "Invalid mdnode for annotation");
    assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
    // start index = 1, to skip the global variable key
    // increment = 2, to skip the value for each property-value pairs
    for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
      // property
      const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
      assert(prop && "Annotation property not a string");

      // value
      ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(md->getOperand(i + 1));
      assert(Val && "Value operand not a constant int");

      std::string keyname = prop->getString().str();
      if (retval.find(keyname) != retval.end())
        retval[keyname].push_back(Val->getZExtValue());
      else {
        std::vector<unsigned> tmp;
        tmp.push_back(Val->getZExtValue());
        retval[keyname] = tmp;
      }
    }
  }

  static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) {
    MutexGuard Guard(Lock);
    NamedMDNode *NMD = m->getNamedMetadata("nvvm.annotations");
    if (!NMD)
      return;
    key_val_pair_t tmp;
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
      const MDNode *elem = NMD->getOperand(i);

      GlobalValue *entity =
          mdconst::dyn_extract_or_null<GlobalValue>(elem->getOperand(0));
      // entity may be null due to DCE
      if (!entity)
        continue;
      if (entity != gv)
        continue;

      // accumulate annotations for entity in tmp
      cacheAnnotationFromMD(elem, tmp);
    }

    if (tmp.empty()) // no annotations for this gv
      return;

    if ((*annotationCache).find(m) != (*annotationCache).end())
      (*annotationCache)[m][gv] = std::move(tmp);
    else {
      global_val_annot_t tmp1;
      tmp1[gv] = std::move(tmp);
      (*annotationCache)[m] = std::move(tmp1);
    }
  }

  bool findOneNVVMAnnotation(const GlobalValue *gv, const std::string &prop,
                             unsigned &retval) {
    MutexGuard Guard(Lock);
    const Module *m = gv->getParent();
    if ((*annotationCache).find(m) == (*annotationCache).end())
      cacheAnnotationFromMD(m, gv);
    else if ((*annotationCache)[m].find(gv) == (*annotationCache)[m].end())
      cacheAnnotationFromMD(m, gv);
    if ((*annotationCache)[m][gv].find(prop) == (*annotationCache)[m][gv].end())
      return false;
    retval = (*annotationCache)[m][gv][prop][0];
    return true;
  }

  bool isAKernelFunction(const Function &F) {
    unsigned x = 0;
    bool retval = findOneNVVMAnnotation(&F, "kernel", x);
    if (!retval) {
      // There is no NVVM metadata, check the calling convention
      return F.getCallingConv() == CallingConv::PTX_Kernel;
    }
    return (x == 1);
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
			//if (isFunctionAKernel(F))
			if (isAKernelFunction(*F))
			{
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
    RegisterMyPass(PassManagerBuilder::EP_EnabledOnOptLevel0,registerPass);
static RegisterStandardPasses
    RegisterMyPass2(PassManagerBuilder::EP_OptimizerLast,registerPass);
}




