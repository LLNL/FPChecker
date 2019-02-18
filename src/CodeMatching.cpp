/*
 * CodeMatching.cpp
 *
 *  Created on: Sep 15, 2018
 *      Author: Ignacio Laguna, ilaguna@llnl.gov
 */

#include "CodeMatching.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/ManagedStatic.h"

using namespace CUDAAnalysis;
using namespace llvm;

namespace {
typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;
} // anonymous namespace
static ManagedStatic<per_module_annot_t> annotationCache;
static sys::Mutex Lock;

bool CodeMatching::isUnwantedFunction(Function *f)
{
	bool ret = false;
	if (
			f->getName().str().find("_FPC_INTERRUPT_") != std::string::npos ||
			f->getName().str().find("_FPC_DEVICE_CODE_FUNC_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_CHECK_ADD_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_CHECK_SUB_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_CHECK_MUL_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_CHECK_DIV_") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_CHECK_ADD_") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_CHECK_SUB_") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_CHECK_MUL_") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_CHECK_DIV_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_IS_SUBNORMAL") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_IS_SUBNORMAL") != std::string::npos ||
			f->getName().str().find("_FPC_LEN_") != std::string::npos ||
			f->getName().str().find("_FPC_CPY_") != std::string::npos ||
			f->getName().str().find("_FPC_CAT_") != std::string::npos ||
			f->getName().str().find("_FPC_PRINT_REPORT_LINE_") != std::string::npos ||
			f->getName().str().find("_FPC_PRINT_REPORT_HEADER_") != std::string::npos ||
			f->getName().str().find("_FPC_PRINT_REPORT_ROW_") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_IS_ALMOST_OVERFLOW") != std::string::npos ||
			f->getName().str().find("_FPC_FP32_IS_ALMOST_SUBNORMAL") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_IS_ALMOST_OVERFLOW") != std::string::npos ||
			f->getName().str().find("_FPC_FP64_IS_ALMOST_SUBNORMAL") != std::string::npos ||
			f->getName().str().find("_FPC_WARNING_") != std::string::npos ||
			f->getName().str().find("_FPC_PRINT_ERRORS_") != std::string::npos ||
			f->getName().str().find("_FPC_INC_ERRORS_") != std::string::npos
			)
		ret = true;

	return ret;
}

bool CodeMatching::isMainFunction(Function *f)
{
	return (f->getName().str().compare("main") == 0);
}

static void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval)
{
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

static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv)
{
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
                           unsigned &retval)
{
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

bool CodeMatching::isAKernelFunction(const Function &F)
{
  unsigned x = 0;
  bool retval = findOneNVVMAnnotation(&F, "kernel", x);
  if (!retval) {
    // There is no NVVM metadata, check the calling convention
    return F.getCallingConv() == CallingConv::PTX_Kernel;
  }
  return (x == 1);
}

bool CodeMatching::isDeviceCode(Module *mod)
{
	bool ret = false;
	for (auto f = mod->begin(), e = mod->end(); f != e; ++f)
	{
		// Discard function declarations
		if (f->isDeclaration())
			continue;

		Function *F = &(*f);
    if (F->getName().str().find("_FPC_DEVICE_CODE_FUNC_") != std::string::npos)
    {
    	ret = true;
    	break;
    }
	}

	return ret;
}
