
#include "Instrumentation.h"
#include "Utility.h"
#include "CodeMatching.h"
#include "Logging.h"

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

#ifdef FPC_DEBUG
	Logging::info("Initializing instrumentation");
	std::stringstream out;
	out << "Pointer value (fp32_check_add_function): " << fp32_check_add_function;
	Logging::info(out.str().c_str());
#endif

  // Find instrumentation function
  for(auto F = M->begin(), e = M->end(); F!=e; ++F)
  {
    Function *f = &(*F);
    if (f->getName().str().find("_FPC_FP32_CHECK_ADD_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_CHECK_ADD_");
#endif

    	fp32_check_add_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_add_function->setCallingConv(CallingConv::PTX_Device);
    	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_SUB_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_CHECK_SUB_");
#endif

    	fp32_check_sub_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_sub_function->setCallingConv(CallingConv::PTX_Device);
    	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_MUL_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_CHECK_MUL_");
#endif

    	fp32_check_mul_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_mul_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_DIV_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_CHECK_DIV_");
#endif

    	fp32_check_div_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp32_check_div_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_ADD_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_CHECK_ADD_");
#endif

    	fp64_check_add_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_add_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_SUB_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_CHECK_SUB_");
#endif

    	fp64_check_sub_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_sub_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_MUL_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_CHECK_MUL_");
#endif

    	fp64_check_mul_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_mul_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_CHECK_DIV_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_CHECK_DIV_");
#endif

    	fp64_check_div_function = f;
    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    	fp64_check_div_function->setCallingConv(CallingConv::PTX_Device);
	f->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    }
    else if (f->getName().str().find("_FPC_INTERRUPT_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_INTERRUPT_");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_SUBNORMAL") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_IS_SUBNORMAL");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_SUBNORMAL") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_IS_SUBNORMAL");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_DEVICE_CODE_FUNC_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_DEVICE_CODE_FUNC_");
#endif

    	if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    		f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_WARNING_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_WARNING_");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_ALMOST_OVERFLOW") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_IS_ALMOST_OVERFLOW");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP32_IS_ALMOST_SUBNORMAL") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP32_IS_ALMOST_SUBNORMAL");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_ALMOST_OVERFLOW") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_IS_ALMOST_OVERFLOW");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
    else if (f->getName().str().find("_FPC_FP64_IS_ALMOST_SUBNORMAL") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_FP64_IS_ALMOST_SUBNORMAL");
#endif

    	//if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    	//	f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
  }

  //printf("Value:  %p\n", fp32_check_add_function);

  // ---- Find instrumentation function in HOST code ----
  for(auto F = M->begin(), e = M->end(); F!=e; ++F)
  {
    Function *f = &(*F);
    if (f->getName().str().find("_FPC_PRINT_AT_MAIN_") != std::string::npos)
    {
#ifdef FPC_DEBUG
    	Logging::info("Found _FPC_PRINT_AT_MAIN_");
#endif
    	print_at_main = f;
    	if (f->getLinkage() != GlobalValue::LinkageTypes::LinkOnceODRLinkage)
    		f->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);
    }
  }
}

void FPInstrumentation::instrumentFunction(Function *f)
{
	if (CodeMatching::isUnwantedFunction(f))
		return;

  assert((fp32_check_add_function!=NULL) && "Function not initialized!");
  assert((fp64_check_add_function!=nullptr) && "Function not initialized!");

#ifdef FPC_DEBUG
	Logging::info("Entering main loop in instrumentFunction");
#endif

	int instrumentedOps = 0;
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
						instrumentedOps++;
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_add_function, args_ref);
						instrumentedOps++;
					}
				}
				else if (inst->getOpcode() == Instruction::FSub)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_sub_function, args_ref);
						instrumentedOps++;
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_sub_function, args_ref);
						instrumentedOps++;
					}
				}
				else if (inst->getOpcode() == Instruction::FMul)
				{
					if (isSingleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp32_check_mul_function, args_ref);
						instrumentedOps++;
					}
					else if (isDoubleFPOperation(inst))
					{
						callInst = builder.CreateCall(fp64_check_mul_function, args_ref);
						instrumentedOps++;
					}
				}
				else if (inst->getOpcode() == Instruction::FDiv)
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

		//Function *F = &(*f);
		BasicBlock *bb = &(f->getEntryBlock());
		inst = bb->getFirstNonPHIOrDbgOrLifetime();
		break;
	}

	assert(inst && "Instruction not valid!");
	return inst;
}

void FPInstrumentation::generateCodeForInterruption()
{

#ifdef FPC_DEBUG
	std::string out = "Generating code for interruption...";
	Logging::info(out.c_str());
#endif

	//for (auto i = mod->global_begin(), end = mod->global_end(); i != end; ++i)
	//{
	//	outs() << "Global: " << *i << "\n";
	//}

	//errs() << "in generteCodeForInterruption\n";
	GlobalVariable *gArray = nullptr;
	gArray = mod->getGlobalVariable ("_ZL15_FPC_FILE_NAME_", true);
	assert((gArray!=nullptr) && "Global array not found");
	//errs() << "setting linkage\n";
	gArray->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
	//errs() << "got array ptr\n";

	std::vector<Constant*> values;
	Instruction *i = firstInstrution();
	//printf("Value:  i: %p\n", i);
	IRBuilder<> builder(i);

	//errs() << "created builder\n";
	std::string fileName = getFileNameFromModule(mod);
	Constant *c = builder.CreateGlobalStringPtr(fileName);
	values.push_back(c);
	//errs() << "pushed value\n";

	ArrayType *t = ArrayType::get(Type::getInt8PtrTy(mod->getContext()), 1);
	//errs() << "got array type\n";
	Constant* init = ConstantArray::get(t, values);

	gArray->setInitializer(NULL);
	//errs() << "Removed existing initializer\n";
	gArray->setInitializer(init);
	//errs() << "initialized\n";
}

void FPInstrumentation::instrumentMainFunction(Function *f)
{
	BasicBlock *bb = &(*(f->begin()));
	Instruction *inst = bb->getFirstNonPHIOrDbg();
	IRBuilder<> builder = createBuilderBefore(inst);
	std::vector<Value *> args;

	CallInst *callInst = nullptr;
	callInst = builder.CreateCall(print_at_main, args);
	assert(callInst && "Invalid call instruction!");
	setFakeDebugLocation(f, callInst);
}

void FPInstrumentation::instrumentErrorArray()
{
	// Global error variable
		GlobalVariable *gArray = nullptr;
		gArray = mod->getGlobalVariable ("_ZL21errors_per_line_array", true);
		//gArray = mod->getNamedGlobal ("errors_per_line_array");
		assert((gArray!=nullptr) && "Global array not found");

		printf("garray type id: %d\n", gArray->getType()->getTypeID());

		if (gArray->getType()->isArrayTy())
			printf("is array\n");
		else if (gArray->getType()->isVectorTy())
			printf("is vector\n");
		else if (gArray->getType()->isIntegerTy())
			printf("is int\n");
		else if (gArray->getType()->isPtrOrPtrVectorTy())
			printf("is ptr or ptr vec\n");
		else if (gArray->getType()->isPointerTy())
			printf("is ptr\n");


		ArrayType *arrType = ArrayType::get(Type::getInt32Ty(mod->getContext()), 77);
		//PointerType* PointerTy_1 = PointerType::get(arrType, 1);
		//Type *ElementType = Type::getInt32PtrTy(mod->getContext());
		//VectorType *vecType =	VectorType::get(ElementType, 77);
		//PointerType *ptrType = 	PointerType::get(Type::getInt32Ty(mod->getContext()), 1);

		GlobalVariable *newGv = nullptr;
		newGv = new GlobalVariable(
				*mod,
				arrType,
				false,
				GlobalValue::LinkageTypes::InternalLinkage, // InternalLinkage,
				0,
				"myVar",
				nullptr,
				GlobalValue::ThreadLocalMode::NotThreadLocal,
				1,
				true
				);

		ConstantAggregateZero* const_array_2 = ConstantAggregateZero::get(arrType);
		newGv->setInitializer(const_array_2);

		//IRBuilder<> builder(mod->getContext());
		//Value *v = builder.CreateBitCast (gArray, arrType, "my");
		//Value *v = builder.CreatePointerBitCastOrAddrSpaceCast(gArray, newGv->getType(), "my");
		//Value *v = builder.CreateBitOrPointerCast(gArray, newGv->getType(), "my");
		//if (v->getType()->isPtrOrPtrVectorTy())
		//			printf("v is ptr or ptr vec\n");
		//printf("v type id: %d\n", v->getType()->getTypeID());
		//gArray->replaceAllUsesWith(v);

    //llvm::Constant *NewPtrForOldDecl =
    //    llvm::ConstantExpr::getBitCast(gArray, newGv->getType());
    //gArray->replaceAllUsesWith(NewPtrForOldDecl);

		for (auto f=mod->begin(), mend=mod->end(); f != mend; ++f)
		{
			for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
			{
				for (auto i=bb->begin(), bend=bb->end(); i != bend; ++i)
				{
					Instruction *inst = &(*i);
					if (CallInst *callInst = dyn_cast<CallInst>(inst))
					{
						std::string instName = inst2str(inst);
						if (instName.find("_ZL9atomicAddPii") != std::string::npos)
						{
							auto pType = PointerType::get(arrType, 0);
							auto addCast = new AddrSpaceCastInst(newGv, pType, "my", inst);
							outs() << "adding: " << inst2str(addCast) << "\n";
							Value* indexList[2] = {ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0), ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0)};
							auto gep = GetElementPtrInst::Create (arrType, addCast, ArrayRef<Value*>(indexList, 2), "my", inst);
							outs() << "gep: " << inst2str(gep) << "\n";

							callInst->setOperand(0, gep);

							//IRBuilder<> builder = createBuilderBefore(inst);
							//builder.CreatePointerBitCastOrAddrSpaceCast(newGv, Type::getInt32PtrTy(mod->getContext(),0), "my");
						}
					}
				}
			}
		}


}
