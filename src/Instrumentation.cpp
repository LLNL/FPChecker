
#include "Instrumentation.h"
#include "Utility.h"
#include "CodeMatching.h"

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

#include <list>

using namespace CUDAAnalysis;
using namespace llvm;

FPInstrumentation::FPInstrumentation(Module *M) :
		mod(M),
		fp32_check_add_function(nullptr),
		fp32_check_sub_function(nullptr),
		fp32_check_mul_function(nullptr),
		fp32_check_div_function(nullptr),
		fp64_check_add_function(nullptr),
		fp64_check_sub_function(nullptr),
		fp64_check_mul_function(nullptr),
		fp64_check_div_function(nullptr)
{

	outs() << "Initializing FPInstrumentation\n";
	printf("Value:  %p\n", fp32_check_add_function);

  // Find instrumentation function
  for(auto F = M->begin(), e = M->end(); F!=e; ++F)
  {
    Function *f = &(*F);
    if (f->getName().str().find("_FPC_FP32_CHECK_ADD_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_CHECK_ADD_\n";
    	fp32_check_add_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_add_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_SUB_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_CHECK_SUB_\n";
    	fp32_check_sub_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_sub_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_MUL_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_CHECK_MUL_\n";
    	fp32_check_mul_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_mul_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_DIV_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_CHECK_DIV_\n";
    	fp32_check_div_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_div_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_ADD_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_CHECK_ADD_\n";
    	fp64_check_add_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_add_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_SUB_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_CHECK_SUB_\n";
    	fp64_check_sub_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_sub_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_MUL_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_CHECK_MUL_\n";
    	fp64_check_mul_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_mul_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_DIV_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_CHECK_DIV_\n";
    	fp64_check_div_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_div_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_INTERRUPT_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_INTERRUPT_\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_SUBNORMAL") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_IS_SUBNORMAL\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_SUBNORMAL") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_IS_SUBNORMAL\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_DEVICE_CODE_FUNC_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_DEVICE_CODE_FUNC_\n";
    	if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    		f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_WARNING_") != std::string::npos)
    {
    	outs() << "====> Found _FPC_WARNING_\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_ALMOST_OVERFLOW") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_IS_ALMOST_OVERFLOW\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_ALMOST_SUBNORMAL") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP32_IS_ALMOST_SUBNORMAL\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_ALMOST_OVERFLOW") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_IS_ALMOST_OVERFLOW\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_ALMOST_SUBNORMAL") != std::string::npos)
    {
    	outs() << "====> Found _FPC_FP64_IS_ALMOST_SUBNORMAL\n";
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
  }

  printf("Value:  %p\n", fp32_check_add_function);
}

void FPInstrumentation::instrumentFunction(Function *f)
{
	if (CodeMatching::isUnwantedFunction(f))
		return;

  assert((fp32_check_add_function!=NULL) && "Function not initialized!");
  assert((fp64_check_add_function!=nullptr) && "Function not initialized!");

	outs() << "Entering main loop in instrumentFunction\n";

	for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
	{
		for (auto i=bb->begin(), bend=bb->end(); i != bend; ++i)
		{
			Instruction *inst = &(*i);

			if (isFPOperation(inst))
			{
				DebugLoc loc = inst->getDebugLoc();
				IRBuilder<> builder = createBuilderAfter(inst);

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

				ArrayRef<Value *> args_ref(args);

				CallInst *callInst = nullptr;

				if (inst->getOpcode() == Instruction::FAdd)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_add_function, args_ref);
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_add_function, args_ref);
					}
				}
				else if (inst->getOpcode() == Instruction::FSub)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_sub_function, args_ref);
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_sub_function, args_ref);
					}
				}
				else if (inst->getOpcode() == Instruction::FMul)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_mul_function, args_ref);
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_mul_function, args_ref);
					}
				}
				else if (inst->getOpcode() == Instruction::FDiv)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_div_function, args_ref);
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_div_function, args_ref);
					}
				}

				assert(callInst && "Invalid call instruction!");
				setFakeDebugLocation(f, callInst);
			}
		}
	}

	outs() << "LEaving main loop in instrumentFunction\n";
}

bool FPInstrumentation::isFPOperation(const Instruction *inst)
{
	return (
			(inst->getOpcode() == Instruction::FMul) ||
			(inst->getOpcode() == Instruction::FDiv) ||
			(inst->getOpcode() == Instruction::FAdd) ||
			(inst->getOpcode() == Instruction::FSub)
				 );
}

bool FPInstrumentation::isDoubleFPOperation(const Instruction *inst)
{
	if (!isFPOperation(inst))
		return false;
	return inst->getType()->isDoubleTy();
}

bool FPInstrumentation::isSingleFPOperation(const Instruction *inst)
{
	if (!isFPOperation(inst))
		return false;
	return inst->getType()->isFloatTy();
}

IRBuilder<> FPInstrumentation::createBuilderAfter(Instruction *inst)
{
	// Get next instruction
  BasicBlock::iterator tmpIt(inst);
  tmpIt++;
  Instruction *nextInst = &(*(tmpIt));
  assert(nextInst && "Invalid instruction!");

	IRBuilder<> builder(nextInst);

	return builder;
}

IRBuilder<> FPInstrumentation::createBuilderBefore(Instruction *inst)
{
	IRBuilder<> builder(inst);

	return builder;
}

void FPInstrumentation::setFakeDebugLocation(Function *f, Instruction *inst)
{
	MDNode *node = f->getMetadata(0);
	assert(node && "Invalid node!");
	DebugLoc newLoc = DebugLoc::get(1, 1, node);
	inst->setDebugLoc(newLoc);
}

/* Returns the return instructions of a function */
/*InstSet FPInstrumentation::finalInstrutions(Function *f)
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
Instruction* FPInstrumentation::firstInstrution()
{
	Instruction *inst = nullptr;
	for (auto f = mod->begin(), e =mod->end(); f != e; ++f)
	{
		// Discard function declarations
		if (f->isDeclaration())
			continue;

		Function *F = &(*f);
		BasicBlock *bb = &(f->getEntryBlock());
		inst = bb->getFirstNonPHIOrDbgOrLifetime();
		break;
	}

	assert(inst && "Instruction not valid!");
	return inst;
}

void FPInstrumentation::generateCodeForInterruption()
{

	for (auto i = mod->global_begin(), end = mod->global_end(); i != end; ++i)
	{
	//	outs() << "Global: " << *i << "\n";
	}

	errs() << "in generteCodeForInterruption\n";
	GlobalVariable *gArray = nullptr;
	//gArray = mod->getNamedGlobal("_FPC_FILE_NAME_");
	gArray = mod->getGlobalVariable ("_ZL15_FPC_FILE_NAME_", true);
	assert((gArray!=nullptr) && "Global array not found");
	errs() << "setting linkage\n";
	gArray->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
	//GArray->setAlignment(4);
	errs() << "got array ptr\n";

	std::vector<Constant*> values;
	//Function *tmpF = mod->getFunction("main");
	//printf("Value:  tmpF: %p\n", tmpF);
	//Instruction *i = tmpF->begin()->getFirstNonPHI();
	Instruction *i = firstInstrution();
	printf("Value:  i: %p\n", i);
	IRBuilder<> builder(i);

	errs() << "created builder\n";
	//std::string fileName = getFileNameFromInstruction(i);
	std::string fileName = getFileNameFromModule(mod);
	Constant *c = builder.CreateGlobalStringPtr(fileName);
	values.push_back(c);
	//c = builder.CreateGlobalStringPtr("value_2");
	//values.push_back(c);
	errs() << "pushed value\n";

	ArrayType *t = ArrayType::get(Type::getInt8PtrTy(mod->getContext()), 1);
	errs() << "got array type\n";
	Constant* init = ConstantArray::get(t, values);

	gArray->setInitializer(NULL);
	errs() << "Removed existing initializer\n";
	gArray->setInitializer(init);
	errs() << "initialized\n";
}
