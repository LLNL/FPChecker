
#include "Instrumentation_int.h"
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
		fp32_check_add_function(nullptr),
		fp32_check_sub_function(nullptr),
		fp32_check_mul_function(nullptr),
		fp32_check_div_function(nullptr),
		fp64_check_add_function(nullptr),
		fp64_check_sub_function(nullptr),
		fp64_check_mul_function(nullptr),
		fp64_check_div_function(nullptr),
		_fpc_print_locations_map_(nullptr)
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
    	confFunction(f, &fp32_check_add_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_ADD_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_SUB_") != std::string::npos)
    {
    	confFunction(f, &fp32_check_sub_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_SUB_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_MUL_") != std::string::npos)
    {
    	confFunction(f, &fp32_check_mul_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_MUL_");
    }
    else if (f->getName().str().find("_FPC_FP32_CHECK_DIV_") != std::string::npos)
    {
    	confFunction(f, &fp32_check_div_function,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_FP32_CHECK_DIV_");
    }
    else if (f->getName().str().find("_FPC_UNUSED_FUNC_") != std::string::npos)
    {
    	confFunction(f, nullptr,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_UNUSED_FUNC_");
    }
    else if (f->getName().str().find("_FPC_INSERT_LOCATIONS_MAP_") != std::string::npos)
    {
    	confFunction(f, nullptr,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_INSERT_LOCATIONS_MAP_");
    }
    else if (f->getName().str().find("_FPC_PRINT_LOCATIONS_MAP_") != std::string::npos)
    {
    	confFunction(f, &_fpc_print_locations_map_,
    	GlobalValue::LinkageTypes::LinkOnceODRLinkage, "_FPC_PRINT_LOCATIONS_MAP_");
    }
  }

  // Globals initialization
  GlobalVariable *table = nullptr;
  table = mod->getGlobalVariable ("_FPC_LOCATIONS_MAP_", true);
  assert(table && "Invalid table!");
  table->setLinkage(GlobalValue::LinkageTypes::LinkOnceODRLinkage);

}

void IntegerInstrumentation::instrumentFunction(Function *f)
{
	if (CodeMatching::isUnwantedFunction(f))
		return;

  assert((fp32_check_add_function!=NULL) && "Function not initialized!");
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

				outs() << "==> inst: " << inst2str(inst) << "\n";

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

				//errs() << "created builder\n";
				std::string fileName = getFileNameFromModule(mod);
				Constant *c = builder.CreateGlobalStringPtr(fileName);
				//values.push_back(c);
					//errs() << "pushed value\n";

				//ArrayType *t = ArrayType::get(Type::getInt8PtrTy(mod->getContext()), 1);
				//errs() << "got array type\n";
				//Constant* init = ConstantArray::get(t, values);

				fName->setInitializer(NULL);
				//errs() << "Removed existing initializer\n";
				fName->setInitializer(c);
				//errs() << "initialized\n";
				args.push_back(loadInst);

				// Update max location number
				if (lineNumber > maxNumLocations)
					maxNumLocations = lineNumber;

				ArrayRef<Value *> args_ref(args);

				CallInst *callInst = nullptr;

				if (inst->getOpcode() == Instruction::Add)
				{
					outs() << "====> In ADD......\n";
					//callInst = builder.CreateCall(fp32_check_add_function, args_ref);
					instrumentedOps++;
				}
				else if (inst->getOpcode() == Instruction::Sub)
				{
					//callInst = builder.CreateCall(fp32_check_sub_function, args_ref);
					instrumentedOps++;

				}
				else if (inst->getOpcode() == Instruction::Mul)
				{
					//callInst = builder.CreateCall(fp32_check_mul_function, args_ref);
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

				//assert(callInst && "Invalid call instruction!");
				//setFakeDebugLocation(f, callInst);
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

/*void FPInstrumentation::generateCodeForInterruption()
{

#ifdef FPC_DEBUG
	std::string out = "Generating code for interruption...";
	Logging::info(out.c_str());
#endif

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
*/


void IntegerInstrumentation::instrumentMainFunction(Function *f)
{
	/*BasicBlock *bb = &(*(f->begin()));
	Instruction *inst = bb->getFirstNonPHIOrDbg();
	IRBuilder<> builder = createBuilderBefore(inst);
	std::vector<Value *> args;

	CallInst *callInst = nullptr;
	callInst = builder.CreateCall(print_at_main, args);
	assert(callInst && "Invalid call instruction!");
	setFakeDebugLocation(f, callInst);*/

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
				auto callInst = builder.CreateCall(_fpc_print_locations_map_, args_ref);
				assert(callInst && "Invalid call instruction!");
			}
		}
	}
}



// Generates a global array on n integers, initialized to zero
/*GlobalVariable* FPInstrumentation::generateIntArrayGlobalVariable(ArrayType *arrType)
{
	GlobalVariable *newGv = nullptr;
	newGv = new GlobalVariable(*mod, arrType, false,
				GlobalValue::LinkageTypes::InternalLinkage, 0,"myVar",
				nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 1, true);

	ConstantAggregateZero* const_array = ConstantAggregateZero::get(arrType);
	newGv->setInitializer(const_array);

	return newGv;
}
*/

/*
void FPInstrumentation::createReadFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName)
{
	for (auto f = mod->begin(), fend = mod->end(); f != fend; ++f)
	{
		if (f->getName().str().find(funcName) != std::string::npos)
		{
#ifdef FPC_DEBUG
			std::string out = "found function: " + funcName;
			Logging::info(out.c_str());
#endif

			// Find return instruction (last instruction)
			// We only have a single basic block
			Instruction *retInst = &(f->begin()->back());
			assert(isa<ReturnInst>(retInst) && "Not a return instruction");

			// Get instruction before the return
			BasicBlock::iterator tmpIt(retInst);
			tmpIt--;
			Instruction *prevInst = &(*(tmpIt));
			assert(prevInst && "Invalid instruction!");

			IRBuilder<> builder = createBuilderBefore(retInst);

			// Create signed extension of parameter
			auto arg = f->arg_begin();
			//auto sext = builder.CreateSExt(arg, Type::getInt64Ty(mod->getContext()), "my");

			// Create GEP inst and addr-space cast inst
			std::vector<Value *> args;
			args.push_back(ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0));
			args.push_back(arg);
			ArrayRef<Value *> indexList(args);
			auto gep = builder.CreateInBoundsGEP(arrType, arr, indexList, "my");
			auto addCast = new AddrSpaceCastInst(gep, Type::getInt64PtrTy(mod->getContext(), 0), "my", retInst);
			auto loadInst = builder.CreateAlignedLoad (addCast, 4, "my");
			retInst->setOperand(0, loadInst);

			// Now we remove old (unused) instructions
			auto iter = f->begin()->begin();
			Instruction *old = &(*iter);
			std::list<Instruction *> iList;
			while (old != prevInst)
			{
				iList.push_back(old);
				iter++;
				old = &(*iter);
			}
			iList.push_back(prevInst);

			for (std::list<Instruction *>::reverse_iterator rit=iList.rbegin(); rit!=iList.rend(); ++rit)
			{
				//outs() << "removing: " << inst2str(*rit) << "\n";
			  (*rit)->eraseFromParent();
			}

			break;
		}
	}
}

void FPInstrumentation::createWriteFunctionForGlobalArray(GlobalVariable *arr, ArrayType *arrType, std::string funcName)
{
	for (auto f = mod->begin(), fend = mod->end(); f != fend; ++f)
	{
		if (f->getName().str().find(funcName) != std::string::npos)
		{
#ifdef FPC_DEBUG
			std::string out = "found function: " + funcName;
			Logging::info(out.c_str());
#endif

			// Find return instruction (last instruction)
			// We only have a single basic block
			Instruction *retInst = &(f->begin()->back());
			assert(isa<ReturnInst>(retInst) && "Not a return instruction");

			// Get instruction before the return
			BasicBlock::iterator tmpIt(retInst);
			tmpIt--;
			Instruction *prevInst = &(*(tmpIt));
			assert(prevInst && "Invalid instruction!");

			IRBuilder<> builder = createBuilderBefore(retInst);

			// Create signed extension of parameter
			auto arg = f->arg_begin();
			//auto sext = builder.CreateSExt(arg, Type::getInt64Ty(mod->getContext()), "my");

			// Create GEP inst and addr-space cast inst
			std::vector<Value *> args;
			args.push_back(ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0));
			args.push_back(arg);
			ArrayRef<Value *> indexList(args);
			auto gep = builder.CreateInBoundsGEP(arrType, arr, indexList, "my");
			auto addCast = new AddrSpaceCastInst(gep, Type::getInt64PtrTy(mod->getContext(), 0), "my", retInst);
			arg++;
			Value *val = &(*arg);
			builder.CreateAlignedStore(val, addCast, 8, false);
			//auto loadInst = builder.CreateAlignedLoad (addCast, 4, "my");

			// Now we remove old (unused) instructions
			auto iter = f->begin()->begin();
			Instruction *old = &(*iter);
			std::list<Instruction *> iList;
			while (old != prevInst)
			{
				iList.push_back(old);
				iter++;
				old = &(*iter);
			}
			iList.push_back(prevInst);

			for (std::list<Instruction *>::reverse_iterator rit=iList.rbegin(); rit!=iList.rend(); ++rit)
			{
				//outs() << "removing: " << inst2str(*rit) << "\n";
			  (*rit)->eraseFromParent();
			}

			break;
		}
	}

}
*/

/*
void FPInstrumentation::instrumentErrorArray()
{
	// Set size of the global array
	int extra = 10;
	int elems = maxNumLocations + extra;

	// Modify initializer of array size for global error array
	GlobalVariable *arrSize = nullptr;
	//arrSize = mod->getGlobalVariable ("_ZL17errors_array_size", true);
	arrSize = mod->getGlobalVariable("_ZL23_FPC_ERRORS_ARRAY_SIZE_", true);
	assert((arrSize!=nullptr) && "Global array not found");
	auto constSize = ConstantInt::get (Type::getInt64Ty(mod->getContext()), (uint64_t)elems, true);
	arrSize->setInitializer(constSize);


	// --- Modify begin of INTERRUPT runtime function -------------------------
	ArrayType *arrType = ArrayType::get(Type::getInt64Ty(mod->getContext()), elems);
	GlobalVariable *newGv = generateIntArrayGlobalVariable(arrType);
#ifdef FPC_DEBUG
        Logging::info("Global errors array created");
#endif

	auto bb = _fpc_interrupt_->begin();
	Instruction *inst = &(*(bb->getFirstNonPHIOrDbg()));
	IRBuilder<> builder = createBuilderBefore(inst);

	auto arg = _fpc_interrupt_->arg_begin();
	arg++; arg++; // get third arg
	auto sext = builder.CreateSExt(arg, Type::getInt64Ty(mod->getContext()), "my");

	std::vector<Value *> args;
	args.push_back(ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0));
	args.push_back(sext);
	ArrayRef<Value *> indexList(args);

	auto gep = builder.CreateInBoundsGEP(arrType, newGv, indexList, "my");
	auto addCast = new AddrSpaceCastInst(gep, Type::getInt64PtrTy(mod->getContext(), 0), "my", inst);

	auto errType = _fpc_interrupt_->arg_begin();
	// subtract 1 from the error type
	auto ext = builder.CreateSExt(errType, Type::getInt64Ty(mod->getContext()), "my");
	Value *subInst = builder.CreateSub (ext, ConstantInt::get(Type::getInt64Ty(mod->getContext()), 3), "my", false, false);

	AtomicCmpXchgInst *cmpXchg = builder.CreateAtomicCmpXchg(
			addCast,
			ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0),
			subInst,
			AtomicOrdering::SequentiallyConsistent,
			AtomicOrdering::SequentiallyConsistent,
			SyncScope::System);
#ifdef FPC_DEBUG
	std::string out = "cmpxchg " + inst2str(cmpXchg) + " created";
	Logging::info(out.c_str());
#endif

	// ----------- Instrument _FPC_READ_GLOBAL_ERRORS_ARRAY -------------
	createReadFunctionForGlobalArray(newGv, arrType, "_FPC_READ_GLOBAL_ERRORS_ARRAY_");
	// ------------------------------------------------------------------------
	// ----------- Instrument _FPC_WRITE_GLOBAL_ERRORS_ARRAY -------------
	createWriteFunctionForGlobalArray(newGv, arrType, "_FPC_WRITE_GLOBAL_ERRORS_ARRAY_");
	// ------------------------------------------------------------------------

	// ============= Instrumentation for Warning function ======================
	GlobalVariable *newWarningsGv = generateIntArrayGlobalVariable(arrType);

	auto bbTmp = _fpc_warning_->begin();
	Instruction *firstInst = &(*(bbTmp->getFirstNonPHIOrDbg()));
	IRBuilder<> builderTmp = createBuilderBefore(firstInst);

	auto argTmp = _fpc_warning_->arg_begin();
	argTmp++; argTmp++; argTmp++; argTmp++; // get 5th arg
	auto bitcastTmp = builderTmp.CreateBitCast (argTmp, Type::getInt64Ty(mod->getContext()), "my");

	auto argTmp2 = _fpc_warning_->arg_begin();
	argTmp2++; argTmp2++; // get third arg
	auto sextTmp = builderTmp.CreateSExt(argTmp2, Type::getInt64Ty(mod->getContext()), "my");

	std::vector<Value *> argsTmp;
	argsTmp.push_back(ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0));
	argsTmp.push_back(sextTmp);
	ArrayRef<Value *> indexTmp(argsTmp);

	auto gepTmp = builderTmp.CreateInBoundsGEP(arrType, newWarningsGv, indexTmp, "my");
	auto addCastTmp = new AddrSpaceCastInst(gepTmp, Type::getInt64PtrTy(mod->getContext(), 0), "my", firstInst);

	AtomicCmpXchgInst *cmpX = builderTmp.CreateAtomicCmpXchg(
			addCastTmp,
			ConstantInt::get(Type::getInt64Ty(mod->getContext()), 0),
			bitcastTmp,
			AtomicOrdering::SequentiallyConsistent,
			AtomicOrdering::SequentiallyConsistent,
			SyncScope::System);
#ifdef FPC_DEBUG
	std::string out2 = "cmpxchg " + inst2str(cmpX) + " created";
	Logging::info(out2.c_str());
#endif

	// ----------- Instrument _Z28_FPC_READ_FP64_GLOBAL_ARRAY_Pyi -------------
	createReadFunctionForGlobalArray(newWarningsGv, arrType, "_FPC_READ_FP64_GLOBAL_ARRAY_");
	// ------------------------------------------------------------------------

	// ----------- Instrument _Z31_FPC_WRITE_GLOBAL_ERRORS_ARRAY_ii -------------
	createWriteFunctionForGlobalArray(newWarningsGv, arrType, "_FPC_WRITE_FP64_GLOBAL_ARRAY_");
	// ------------------------------------------------------------------------
}

void FPInstrumentation::instrumentEndOfKernel(Function *f)
{
	// Find the return instructions
	for (auto bb=f->begin(), end=f->end(); bb != end; ++bb)
	{
		for (auto i=bb->begin(), iend=bb->end(); i != iend; ++i)
		{
			Instruction *inst = &(*i);
			if (isa<ReturnInst>(inst))
			{
				IRBuilder<> builder = createBuilderBefore(inst);
				std::vector<Value *> args;
				CallInst *callInst = builder.CreateCall(_fpc_print_errors_, args);
				assert(callInst && "Invalid call instruction!");
				setFakeDebugLocation(f, callInst);
			}
		}
	}
}
*/
