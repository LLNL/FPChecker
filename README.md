# FPChecker

**FPChecker** (or Floating-Point Checker) is a compiler framework to check for floating-point exceptions in CUDA. It is designed as a Clang/LLVM extension that instruments CUDA code to catch floating-point exceptions at runtime.

## Detectable Errors and Warnings
FPChecker detects floating-point computations that produce:
- Overflows: +INF and -INF values
- Underflows: subnormal (or denormalized) values
- NANs:  coming, for example, from 0/0

When at least one of the threads in a CUDA grid produces any of the above cases, an error report is generated.

FPChecker also generates **warning reports** for computations that are very close to become overflows or underflows, i.e., `x%` from the limits of normal values, where `x` is configurable.

# Getting Started

## Building
.... TBD

## Requirements to Use FPChecker
The primary requirement for using FPChecker is to be able to compile your CUDA code with a recent version of the clang/LLVM compiler. Pure CUDA code or RAJA (with CUDA execution) are supported.

We have tested FPChecker so far with these versions of clang/LLVM:
- clang 7.x
- clang 8.0

For more information about compiling CUDA with clang, plese refer to [Compiling CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html).

## Using FPChecker
Once you are able to compile and run your CUDA application with clang, follow this to enable FPChecker:

1. Add this to your Makefile:

```sh
FPCHECKER_PATH  = /path/to/fpchecker
LLVM_PASS       = -Xclang -load -Xclang $(FPCHECKER_PATH)/lib/libcudakernels.so \
-include Runtime.h -I$(FPCHECKER_PATH)/runtime
CXXFLAGS += $(LLVM_PASS)
```

This will tell clang where the FPChecker runtime is located. FPCHECKER_PATH is the where FPChecker is installed.

2. Compile your code and run it.

##Expected Output

When your applications begins to run, you should see the following, indicating that your application was instrumented by FPChecker:

```sh
========================================
 FPChecker (v0.0.3, Feb 21 2019)
========================================
```

If an exception is found, your kernel will be aborted and an error report like the following will be shown:
```sh
+--------------------------- FPChecker Error Report ---------------------------+
 Error         : Underflow                                                     
 Operation     : MUL (9.999888672e-321)                                            
 File          : dot_product_raja.cpp                                          
 Line          : 32                                                            
+------------------------------------------------------------------------------+
```

## MPI Awareness
The current version is not MPI aware, so every MPI process that encounters an error/warning will print a report. Future versions will allow only a single MPI process rank (e.g., rank 0) to print reports.

# Configuration Options
Configuration options are passed via -D macros when invoking clang to compile your code.

- **-D FPC_DANGER_ZONE_PERCENT=x.x:** Changes the size of the danger zone. By default, x.x is 0.10, and it should be a number between 0.0 and 1.0. Warning reports can be almost completely disabled by using a small danger zone, such as 0.01.
- **-D FPC_ERRORS_DONT_ABORT:** By default FPChecker aborts the kernel that first encounters an error (i.e., floating-point exception). Depending on your application, this may make the host code abort as well. This option allows FPChecker to print error/warning reports without aborting. This allows you to check for errors/warnings in the entire execution of your program. The performance of this mode is not as good as in the default mode.

### Contact
For questions, contact Ignacio Laguna <ilaguna@llnl.gov>.
