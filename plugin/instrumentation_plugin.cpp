//===- clang_plugin.cpp ---------------------------------------------===//
//
// This clang plug-in instruments the source code.
// For each expression E that evaluates to a floating-point type,
// we insert a callback function to the runtime that uses E as argument.
//
// For example:
//    double x = a[i] + 3.1415;
// is transformed to:
//    double x = _FPC_CHECK_(a[i] + 3.1415);
//
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ParentMap.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <set>

using namespace clang;

namespace CUDAAnalysis {

Rewriter *rewriter;
std::string globalFileName("none");

class InstrumentFunctionVisitor
    : public RecursiveASTVisitor<InstrumentFunctionVisitor> {

private:
  ASTContext *Context;

  typedef std::pair<std::string, unsigned> PairLocation;
  std::set<PairLocation> instrumentedCode;

public:
  explicit InstrumentFunctionVisitor(ASTContext *Context) : Context(Context) {}

  std::string getStmtString(const Stmt *stmt) {
    std::string s;
    llvm::raw_string_ostream stream(s);
    stmt->printPretty(stream, NULL, PrintingPolicy(LangOptions()));
    return stream.str();
  }

  std::string printInstrumentedLocation(const BinaryOperator *bo) {
    std::string ret;
    // SourceLocation sl = bo->getEndLoc();
    SourceLocation sl = bo->getEndLoc();
    FullSourceLoc fullLocation = Context->getFullLoc(sl);
    if (fullLocation.isValid()) {
      // Get file
      std::string fileName = Context->getSourceManager().getFilename(sl).str();
      llvm::outs() << "#FPCHECKER: instrumenting " << fileName
                   << " @ line: " << fullLocation.getSpellingLineNumber()
                   << "\n";
      // llvm::outs() << " globalFileName: " << globalFileName << "\n";
    }
    return ret;
  }

  std::string getLineNumber(const BinaryOperator *bo) {
    std::string ret("0");
    SourceLocation sl = bo->getEndLoc();
    FullSourceLoc fullLocation = Context->getFullLoc(sl);
    if (fullLocation.isValid()) {
      ret = std::to_string(fullLocation.getSpellingLineNumber());
    }
    return ret;
  }

  std::string getShortFileName()
  {
    std::string ret("unknown");
    unsigned n=25;
    if (globalFileName.size() <= n)
      ret = globalFileName;
    else
      ret = ".."+globalFileName.substr(globalFileName.size() - n);
    return ret;
  }

  void printLineAndColumn(const BinaryOperator *bo) {
    SourceLocation sl = bo->getEndLoc();
    FullSourceLoc fullLocation = Context->getFullLoc(sl);
    llvm::outs() << "\t **** " << getStmtString(bo) << ": line: " <<
        fullLocation.getSpellingLineNumber() << " col: " <<
        fullLocation.getColumnNumber() << "\n";
  }

  // This function checks if this block has been previously instrumented
  // It gets the parent statement, and checks if there is a call statement
  // with a runtime call.
  /*bool hasBeenInstrumented(const BinaryOperator *E) {
    // Get parents of expression and iterate on all children
    llvm::outs() << "Reg: " << getStmtString(E) << "\n";
    auto nodeList = Context->getParents(*E);
    for (size_t i = 0; i < nodeList.size(); ++i) {
      llvm::outs() << "Parent: "
                   << nodeList[i].getNodeKind().asStringRef().str() << "\n";
      const Stmt *pstmt = nodeList[i].get<Stmt>();
      if (pstmt) {
        // llvm::outs() << "Stmt: " << getStmtString(pstmt) << "\n";
        for (Stmt::const_child_iterator j = pstmt->child_begin(),
                                        e = pstmt->child_end();
             j != e; ++j) {
          const Stmt *currStmt = *j;
          // llvm::outs() << "Child: " << getStmtString(currStmt);
          if (const CallExpr *call = dyn_cast<CallExpr>(currStmt)) {
            std::string str = getStmtString(call);
            // llvm::outs() << "\nCall expr: " << str << "\n";
            if (str.find("_FPC_CHECK_") != std::string::npos)
              return true;
          }
        }
      }
    }
    return false;
  }*/

  // Check if a function body has been instrumented
  bool hasBeenInstrumented(const Stmt *stmt) {
    if (stmt == nullptr)
      return false;

    bool ret = false;
    for (Stmt::const_child_iterator j = stmt->child_begin(),
         e = stmt->child_end(); j != e; ++j) {
      const Stmt *child = *j;
      if (child == nullptr)
        continue; // continue with next child

      if (const CallExpr *call = dyn_cast<CallExpr>(child)) {
        std::string str = getStmtString(call);
        if (str.find("_FPC_CHECK_") != std::string::npos) {
          return true;
        } else {
          ret = hasBeenInstrumented(child);
          if (ret) {
            break;
          }
        }
      } else {
        ret = hasBeenInstrumented(child);
        if (ret)
          break;
      }
    }
    return ret;
  }

  /// check if this is a call to the runtime, i.e., an instrumented expr
  //bool isFPCheckerCall(const BinaryOperator *bo) {
  //}

  /// Check if the parent is a compound statement
  /*bool isParentCompoundStatement(const Stmt *s) {
    auto nodeList = Context->getParents(*s);
    for (size_t i = 0; i < nodeList.size(); ++i) {
      const Stmt *compoundStmt = nodeList[i].get<CompoundStmt>();
      if (compoundStmt)
        return true;
    }
    return false;
  }*/

  enum PluginCUDAFunctionTarget {
    CFT_Device,
    CFT_Global,
    CFT_Host,
    CFT_HostDevice,
    CFT_InvalidTarget
  };

  /// IdentifyCUDATarget - Determine the CUDA compilation target for this
  /// function
  PluginCUDAFunctionTarget
  IdentifyCUDATarget(const FunctionDecl *D, bool IgnoreImplicitHDAttr = true) {
    // Code that lives outside a function is run on the host.
    if (D == nullptr)
      return CFT_Host;

    if (D->hasAttr<CUDAInvalidTargetAttr>())
      return CFT_InvalidTarget;

    if (D->hasAttr<CUDAGlobalAttr>())
      return CFT_Global;

    if (D->hasAttr<CUDADeviceAttr>()) {
      if (D->hasAttr<CUDAHostAttr>())
        return CFT_HostDevice;
      return CFT_Device;
    } else if (D->hasAttr<CUDAHostAttr>()) {
      return CFT_Host;
    } else if (D->isImplicit() && !IgnoreImplicitHDAttr) {
      // Some implicit declarations (like intrinsic functions) are not marked.
      // Set the most lenient target on them for maximal flexibility.
      return CFT_HostDevice;
    }

    return CFT_Host;
  }

//  void rewriteCode(SourceRange &range, std::string &txt, const BinaryOperator *bo) {
  void rewriteCode(SourceRange &range, std::string &leftTxt, std::string &rightTxt, const BinaryOperator *bo) {
    SourceLocation sl = bo->getEndLoc();
    FullSourceLoc fullLocation = Context->getFullLoc(sl);
    unsigned line = 0;
    if (fullLocation.isValid()) {
      line = fullLocation.getSpellingLineNumber();
    }
    std::string expr = getStmtString(bo);
    PairLocation loc(expr, line);
    auto l = range.getBegin();
    auto r = range.getEnd();
    if (!Rewriter::isRewritable(l) || !Rewriter::isRewritable(r))
      return; // don't do anything if not rewritable

    /// check if we have instrumented the location before
    if ( instrumentedCode.find(loc) == instrumentedCode.end()) {
      //rewriter->ReplaceText(range, txt);
      if (rewriter->InsertTextBefore(l, leftTxt))
        llvm::outs() << "#FPCHECKER: left location not rewritable\n";
      if (rewriter->InsertTextAfterToken(r, rightTxt))
        llvm::outs() << "#FPCHECKER: right location not rewritable\n";
      instrumentedCode.insert(loc);
    } else {
      llvm::errs() << "FPChecker warning: instrumenting repeated location: " << expr << "\n";
      //exit(EXIT_FAILURE);
    }
  }

  /*void removeUnwantedString(std::string &s) {
    if (s.rfind("class ", 0) == 0) {
      s.erase(0, 6);
    } else if (s.rfind("this->class ", 0) == 0) {
      s.erase(0, 12);
    }
  }*/

  /// Analyze BinaryOperator and instrument it.
  /// Return true if we can instrument; otherwise return false
  bool AnalyzetBinaryOperator(const BinaryOperator *E) {
    // Do not instrument ASTs from different files (e.g., header files)
    std::string fileNameOfExpression =
        Context->getSourceManager().getFilename(E->getEndLoc()).str();
    if (fileNameOfExpression.compare(globalFileName) != 0)
      return false;

    // Only instrument in CUDA device path
    const LangOptions &langOptions = Context->getLangOpts();
    if (!langOptions.CUDAIsDevice)
      return false;

    bool ret = false;
    // Verify the operation evaluates to floating-point
    if (const Type *type = E->getType().getTypePtr()) {
      if (type->isFloatingType()) {
        Expr *lhs = E->getLHS();
        Expr *rhs = E->getRHS();
        std::string lineNumber = getLineNumber(E);
        if (E->getOpcode() == BO_AddAssign || E->getOpcode() == BO_SubAssign ||
            E->getOpcode() == BO_MulAssign || E->getOpcode() == BO_DivAssign ||
            E->getOpcode() == BO_Assign) {
#ifdef FPC_DEBUG
          printInstrumentedLocation(E);
#endif
          /*std::string prettyRHS;
          if (auto *callExpr = dyn_cast<CallExpr>(rhs)) {
            prettyRHS = getStmtString(callExpr);
            removeUnwantedString(prettyRHS);
          } else {
            prettyRHS = getStmtString(rhs);
          }*/
          //std::string txt("_FPC_CHECK_(" + getStmtString(rhs) +", "+lineNumber+", \""+getShortFileName()+"\"" + ")");
          //std::string txt("_FPC_CHECK_(" + prettyRHS +", "+lineNumber+", \""+getShortFileName()+"\"" + ")");
          std::string leftTxt("_FPC_CHECK_(");
          std::string rightTxt(", "+lineNumber+", \""+getShortFileName()+"\""+")");
          SourceRange range(rhs->getBeginLoc(), rhs->getEndLoc());
          rewriteCode(range, leftTxt, rightTxt, E);
          //rewriteCode(range, txt, E);
          ret = true;
          //}
        } else if(E->getOpcode() == BO_Add || E->getOpcode() == BO_Sub ||
            E->getOpcode() == BO_Mul || E->getOpcode() == BO_Div) { /// it's not an assignment
#ifdef FPC_DEBUG
          printInstrumentedLocation(E);
#endif
          //std::string txt("_FPC_CHECK_(" + getStmtString(E) +", "+lineNumber+", \""+getShortFileName()+"\"" + ")");
          SourceRange range(lhs->getBeginLoc(), rhs->getEndLoc());
          std::string leftTxt("_FPC_CHECK_(");
          std::string rightTxt(", "+lineNumber+", \""+getShortFileName()+"\""+")");
          rewriteCode(range, leftTxt, rightTxt, E);
          //rewriteCode(range, txt, E);
          ret = true;
        }
      }
    }

    return ret;
  }

  void recursiveVisit(const Stmt *stmt) {
    for (Stmt::const_child_iterator j = stmt->child_begin(),
                                    e = stmt->child_end(); j != e; ++j) {
      const Stmt *child = *j;
      if (child) {
        if (const BinaryOperator *bo = dyn_cast<BinaryOperator>(child)) {
          if (!AnalyzetBinaryOperator(bo)) {
            /// keep visiting if we cannot instrument
            recursiveVisit(child);
          }
        } else {
          recursiveVisit(child);
        }
      }
    }
  }

  /// This is the clang visitor class
  bool VisitFunctionDecl(const FunctionDecl *funcDecl) {
    if (funcDecl->hasBody(funcDecl)) {
      if (Stmt *stmt = funcDecl->getBody(funcDecl)) {
        //auto target = IdentifyCUDATarget(funcDecl, true);
        //if (target == CFT_Device || target == CFT_HostDevice || target == CFT_Global) {
          /// Instrument  only if the body has not been instrumented
          if (!hasBeenInstrumented(stmt)) {
            recursiveVisit(stmt);
          }
        //}
      }
    }
    return true;
  }
};

class FindNamedClassConsumer : public clang::ASTConsumer {
public:
  explicit FindNamedClassConsumer(ASTContext *Context) : Visitor(Context) {}

  // Used by std::find_if
  struct MatchPathSeparator {
    bool operator()(char ch) const { return ch == '/'; }
  };

  std::string basename(std::string path) {
    return std::string(
        std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(),
        path.end());
  }

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    rewriter = new Rewriter(Context.getSourceManager(), Context.getLangOpts());
    // Create an output file to write the updated code
    FileID id = rewriter->getSourceMgr().getMainFileID();
    std::string
        filename = // "/tmp/" +
                   // basename(rewriter->getSourceMgr().getFilename(rewriter->getSourceMgr().getLocForStartOfFile(id)).str());
        rewriter->getSourceMgr()
            .getFilename(rewriter->getSourceMgr().getLocForStartOfFile(id))
            .str();
    globalFileName = filename;

    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
#ifdef FPC_DEBUG
    llvm::outs() << "#FPCHECKER: Done analyzing file ***\n";
#endif

    std::error_code OutErrorInfo;
    std::error_code ok;
    // llvm::raw_fd_ostream outFile(llvm::StringRef(filename), OutErrorInfo,
    // llvm::sys::fs::F_None);
    // Get ostream for tmp file
    llvm::raw_fd_ostream outFile(llvm::StringRef(filename + ".tmp"),
                                 OutErrorInfo, llvm::sys::fs::F_None);
    if (OutErrorInfo == ok) {
      // Getting a buffer to write the file
      const RewriteBuffer *RewriteBuf = rewriter->getRewriteBufferFor(id);
      // assert(RewriteBuf!=nullptr && "got buffer");
      if (RewriteBuf != nullptr) {
        outFile << std::string(RewriteBuf->begin(), RewriteBuf->end());
#ifdef FPC_DEBUG
        llvm::outs() << "#FPCHECKER: Output file created in: " << filename << "\n";
#endif
        // Rename tmp file
        if (std::error_code ec =
                llvm::sys::fs::rename(filename + ".tmp", filename)) {
          llvm::errs() << "#FPCHECKER: Error. Unable to rename temporal file!!\n";
          // If the remove fails, there's not a lot we can do (this is already
          // an error).
        }
      } else {
        //llvm::errs() << "#FPCHECKER: File was unmodified:" << filename << "\n";
      }
      outFile.close();
      // Remove tmp file
      llvm::sys::fs::remove(filename + ".tmp");

    } else {
      llvm::errs() << "#FPCHECKER: error - could not create file: (" <<
          OutErrorInfo.category().name() << " error)\n";
    }
  }

private:
  InstrumentFunctionVisitor Visitor;
};

class InstrumentFunctionAction : public PluginASTAction {
  std::set<std::string> ParsedTemplates;

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 llvm::StringRef) override {
    // return std::make_unique<FindNamedClassConsumer>(CI, ParsedTemplates);
    return std::unique_ptr<clang::ASTConsumer>(
        new FindNamedClassConsumer(&CI.getASTContext()));
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    if (!args.empty() && args[0] == "help")
      PrintHelp(llvm::errs());

    return true;
  }
  void PrintHelp(llvm::raw_ostream &ros) {
    ros << "Help for PrintFunctionNames plugin goes here\n";
  }

  PluginASTAction::ActionType getActionType() override {
    return PluginASTAction::ActionType::ReplaceAction;
    // return PluginASTAction::ActionType::AddBeforeMainAction;
    // return PluginASTAction::ActionType::AddAfterMainAction;
  }

  bool BeginInvocation(CompilerInstance &CI) override {
    /*LangOptions &lop = CI.getLangOpts();
     if (lop.CUDA) {
             llvm::outs() << "lop.CUDA\n";
       if (lop.CUDAIsDevice)
       {
           llvm::outs() << "lop.CUDAIsDevice\n";
       }
     }*/

    return true;
  }
};

} // namespace

static FrontendPluginRegistry::Add<CUDAAnalysis::InstrumentFunctionAction>
    X("instrumentation_plugin", "Instrument functions for FPChecker runtime");
