
#include "Utility.h"
#include "CodeMatching.h"
#include "Logging.h"

#include "Instrumentation_int.h"

#include <llvm/IR/Type.h>
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Constants.h"

#include <list>
#include <string>

using namespace CUDAAnalysis;
using namespace llvm;

/* This function configures the function found (e.g., calling conventions) and
saves pointer if needed. We also do logging. */
void confFunction(Function *found, Function **saveHere,
		GlobalValue::LinkageTypes linkage, const char *name)
{
#ifdef FPC_DEBUG
	std::string out = std::string("Found ") + std::string(name);
  Logging::info(out.c_str());
#endif

  if (saveHere != nullptr) // if we want to save the function pointer
  	*saveHere = found;
  //if (found->getCallingConv() != cc)
  //	found->setCallingConv(cc);
  if (found->getLinkage() != linkage)
  	found->setLinkage(linkage);
}

IntegerInstrumentation::IntegerInstrumentation(Module *M) :
		mod(M),
	  int32_check_add_function(nullptr),
	  int32_check_sub_function(nullptr),
	  int32_check_mul_function(nullptr),
	  int32_check_div_function(nullptr),
		_fpc_init_htable_(nullptr),
		_fpc_print_locations_(nullptr)
{

#ifdef FPC_DEBUG
	Logging::info("Initializing *integer* instrumentation");
#endif

  // Find and configure instrumentation functions
  for(auto F = M->begin(), e = M->end(); F!=e; ++F)
  {
    Function *f = &(*F);
    if (f->getName().str().find("_FPC_FP32_CHECK_ADD_") != std::string::npos)
    {
    	confFunction(f, &int32_check_add_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_ADD_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_SUB_") != std::string::npos)
    {
    	confFunction(f, &int32_check_sub_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_SUB_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_MUL_") != std::string::npos)
    {
    	confFunction(f, &int32_check_mul_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_MUL_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_DIV_") != std::string::npos)
    {
    	confFunction(f, &int32_check_div_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_DIV_");
    }
    else if (f->getName().str().find("_FPC_UNUSED_FUNC_") != std::string::npos)
    {
    	confFunction(f, nullptr,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_UNUSED_FUNC_");
    }
    else if (f->getName().str().find("_FPC_INIT_HASH_TABLE_") != std::string::npos)
    {
    	confFunction(f, &_fpc_init_htable_,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_INIT_HASH_TABLE_");
    }
    else if (f->getName().str().find("_FPC_PRINT_LOCATIONS_") != std::string::npos)
    {
    	confFunction(f, &_fpc_print_locations_,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_PRINT_LOCATIONS_");
    }
   else if (f->getName().str().find("_FPC_HT_CREATE_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_HT_CREATE_");
   }
   else if (f->getName().str().find("_FPC_HT_NEWPAIR_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_HT_NEWPAIR_");
   }
   else if (f->getName().str().find("_FPC_HT_SET_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_HT_SET_");
   }
   else if (f->getName().str().find("_FPC_HT_HASH_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_HT_HASH_");
   }
   else if (f->getName().str().find("_FPC_ITEMS_EQUAL_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_ITEMS_EQUAL_");
   }
   else if (f->getName().str().find("_FPC_PRINT_HASH_TABLE_") != std::string::npos)
   {
   	confFunction(f, nullptr,
   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_PRINT_HASH_TABLE_");
   }
	   else if (f->getName().str().find("_FPC_CHECK_OVERFLOW_") != std::string::npos)
	   {
	   	confFunction(f, nullptr,
	   	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_CHECK_OVERFLOW_");
	   }
  }

  // Globals initialization
  GlobalVariable *table = nullptr;
  table = mod->getGlobalVariable ("_FPC_HTABLE_", true);
  assert(table && "Invalid table!");
  table->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
}

void IntegerInstrumentation::instrumentFunction(Function *f)
{
	if (CodeMatching::isUnwantedFunction(f))
		return;

  assert((int32_check_add_function!=NULL) && "Function not initialized!");
  //assert((fp64_check_add_function!=nullptr) && "Function not initialized!");

#ifdef FPC_DEBUG
	Logging::info("Entering main loop in instrumentFunction");
#endif

	int instrumentedOps = 0;
	for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
	{
		for (auto i=bb->begin(), bend=bb->end(); i != bend; ++i)
		{
			Instruction *inst = &(*i);

			if (isIntOperation(inst) && is32BitIntOperation(inst))
			{
				DebugLoc loc = inst->getDebugLoc();
				IRBuilder<> builder = createBuilderAfter(inst);

				//outs() << "==> inst: " << inst2str(inst) << "\n";

				// Push parameters
				std::vector<Value *> args;
				args.push_back(inst);
				args.push_back(inst->getOperand(0));
				args.push_back(inst->getOperand(1));
				//args.push_back(ConstantFP::get(builder.getDoubleTy(), 999.0));

				// Push location parameter (line number)
				int lineNumber = getLineOfCode(inst);
				ConstantInt* locId = ConstantInt::get(mod->getContext(), APInt(32, lineNumber, true));
				args.push_back(locId);

				// Get global fileName pointer
				//errs() << "in generteCodeForInterruption\n";
				GlobalVariable *fName = nullptr;
				fName = mod->getGlobalVariable ("_ZL15_FPC_FILE_NAME_", true);
				assert((fName!=nullptr) && "Global array not found");
				auto loadInst = builder.CreateAlignedLoad (fName, 4, "my");

				//std::string fileName = getFileNameFromModule(mod);
				//std::string fileName = mod->getSourceFileName();
				std::string fileName = getFileNameFromInstruction(inst);
				Constant *c = builder.CreateGlobalStringPtr(fileName);
				fName->setInitializer(NULL);
				fName->setInitializer(c);
				args.push_back(loadInst);

				// Update max location number
				if (lineNumber > maxNumLocations)
					maxNumLocations = lineNumber;

				ArrayRef<Value *> args_ref(args);

				CallInst *callInst = nullptr;

				if (inst->getOpcode() == Instruction::Add)
				{
					callInst = builder.CreateCall(int32_check_add_function, args_ref);
					instrumentedOps++;
				}
				else if (inst->getOpcode() == Instruction::Sub)
				{
					callInst = builder.CreateCall(int32_check_sub_function, args_ref);
					instrumentedOps++;

				}
				else if (inst->getOpcode() == Instruction::Mul)
				{
					callInst = builder.CreateCall(int32_check_mul_function, args_ref);
					instrumentedOps++;
				}
				/*else if (inst->getOpcode() == Instruction::FDiv)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_div_function, args_ref);
						instrumentedOps++;
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_div_function, args_ref);
						instrumentedOps++;
					}
				}
				*/

				assert(callInst && "Invalid call instruction!");
				setFakeDebugLocation(f, callInst);
			}
		}
	}

#ifdef FPC_DEBUG
	std::stringstream out;
	out << "Instrumented operations: " << instrumentedOps;
	Logging::info(out.str().c_str());
	Logging::info("Leaving main loop in instrumentFunction");
#endif
}

bool IntegerInstrumentation::isIntOperation(const Instruction *inst)
{
	return (
			(inst->getOpcode() == Instruction::Mul) ||
			//(inst->getOpcode() == Instruction::SDiv) ||
			(inst->getOpcode() == Instruction::Add) ||
			(inst->getOpcode() == Instruction::Sub)
				 );
}


bool IntegerInstrumentation::is64BitIntOperation(const Instruction *inst)
{
	return inst->getType()->isIntegerTy(64);
}

bool IntegerInstrumentation::is32BitIntOperation(const Instruction *inst)
{
	return inst->getType()->isIntegerTy(32);
}


IRBuilder<> IntegerInstrumentation::createBuilderAfter(Instruction *inst)
{
	// Get next instruction
  BasicBlock::iterator tmpIt(inst);
  tmpIt++;
  Instruction *nextInst = &(*(tmpIt));
  assert(nextInst && "Invalid instruction!");

	IRBuilder<> builder(nextInst);

	return builder;
}

IRBuilder<> IntegerInstrumentation::createBuilderBefore(Instruction *inst)
{
	IRBuilder<> builder(inst);

	return builder;
}

void IntegerInstrumentation::setFakeDebugLocation(Function *f, Instruction *inst)
{
	MDNode *node = f->getMetadata(0);
	assert(node && "Invalid node!");
	DebugLoc newLoc = DebugLoc::get(1, 1, node);
	inst->setDebugLoc(newLoc);
}

/* Returns the return instructions of a function */
/*InstSet IntegerInstrumentation::finalInstrutions(Function *f)
{
	InstSet finalInstructions;
	for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
	{
		for (auto i=bb->begin(), iend=bb->end(); i != iend; ++i)
		{
			Instruction *inst = &(*i);
			if (isa<ReturnInst>(inst))
				finalInstructions.insert(inst);
		}
	}
	return finalInstructions;
}*/

/* Returns the return first (non-phi) instruction of the module */
Instruction* IntegerInstrumentation::firstInstrution()
{
	Instruction *inst = nullptr;
	for (auto f = mod->begin(), e =mod->end(); f != e; ++f)
	{
		// Discard function declarations
		if (f->isDeclaration())
			continue;

		//Function *F = &(*f);
		BasicBlock *bb = &(f->getEntryBlock());
		inst = bb->getFirstNonPHIOrDbgOrLifetime();
		break;
	}

	assert(inst && "Instruction not valid!");
	return inst;
}

void IntegerInstrumentation::instrumentMainFunction(Function *f)
{
	/// ----------------- BEGIN --------------------------
	BasicBlock *bb = &(*(f->begin()));
	Instruction *inst = bb->getFirstNonPHIOrDbg();
	IRBuilder<> builder = createBuilderBefore(inst);
	std::vector<Value *> args;

	CallInst *callInst = nullptr;
	callInst = builder.CreateCall(_fpc_init_htable_, args);
	assert(callInst && "Invalid call instruction!");
	setFakeDebugLocation(f, callInst);


	/// ------------------ END ----------------------------
	/// Print table before end of function
	for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
	{
		for (auto i=bb->begin(), iend=bb->end(); i != iend; ++i)
		{
			Instruction *inst = &(*i);
			if (isa<ReturnInst>(inst))
			{
				std::vector<Value *> args;
				ArrayRef<Value *> args_ref(args);
				IRBuilder<> builder = createBuilderBefore(inst);
				auto callInst = builder.CreateCall(_fpc_print_locations_, args_ref);
				assert(callInst && "Invalid call instruction!");
			}
		}
	}
}
