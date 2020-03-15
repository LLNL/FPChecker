# FPChecker

[![Build Status](https://travis-ci.org/LLNL/FPChecker.svg?branch=master)](https://travis-ci.org/LLNL/FPChecker)

**FPChecker** (or Floating-Point Checker) is a framework to detect floating-point exceptions in CUDA. It is designed as a Clang/LLVM extension that instruments CUDA code to catch floating-point exceptions at runtime.

## Detectable Errors and Warnings
FPChecker detects floating-point computations that produce:
- Overflows: +INF and -INF values
- Underflows: subnormal (or denormalized) values
- NANs:  coming, for example, from 0.0/0.0

When at least one of the threads in a CUDA grid produces any of the above cases, an error report is generated.

FPChecker also generates **warning reports** for computations that are close to become overflows or underflows, i.e., `x%` from the limits of normal values, where `x` is configurable.

# Getting Started

## How to Use FPChecker

FPChecker instruments the CUDA application code. This instrumentation can be executed via the *clang* frontend, or via the *llvm* intermediate representation. We call these two ways of using FPChecker the **Clang version** and the **LLVM version**, respectively.

The **Clang version** instruments the source code of the application using a clang plugin. The instrumentation changes every expression `E` that evaluates to a floating-point value, to `_FPC_CHECK_(E)`. After theses transformations are performed, the code can be compiled with nvcc.

The **LLVM version** on the other hand, performs instrumentation in the LLVM compiler itself (in the intermediate representation, or IR).

Both versions have advantages and disadvantages:
- **Clang version**: the final code can be compiled with nvcc; however, this version can be slower than the LLVM version and requires a two-pass compilation process (i.e., first instrument using clang and then compile/link with nvcc).
- **LLVM version**: it is faster than the Clang version as code instrumented *after* optimizations are applied; however, it requires the application to be  compiled completely using clang (clang does not support the same functionality than nvcc, and some CUDA applications cannot be compiled with clang).

## Building
You can build using `cmake`:
```sh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../
make
make install
```

`cmake` will attempt to search for `clang++` and `llvm-config` in your environment. Make sure these commands are visible.

Optionally you can run tests by running `make tests` after executing `make`. Tests are executed in `python` version 2.7.x, and require the `pytest` module. Also, the environment variable `CUDA_PATH` needs to be set to the location of the CUDA toolkit before running the tests.

## Using the FPChecker Clang Version

Using this version requires following two steps: (1) instrumenting the source code (with a clang plugin) and (2) compiling the code with nvcc.

### Step 1: Instrumenting the source code

This step can be executed either by using the `clang-fpchecker` wrapper script or by directly loading the plugin (the `clang-fpchecker` wrapper automatically calls the required options to load the plugin). We explain both methods as follows.

#### Using the clang-fpchecker Wrapper Script

The `clang-fpchecker` wrapper can be used as if we are using `clang` to compile files. For example, suppose we are instrumenting the `compute.cu` CUDA file; the wrapper is called this way:

```sh
clang-fpchecker --cuda-gpu-arch=sm_60 -x cuda -c compute.cu
```

Note that in clang, the `--cuda-gpu-arch` flag specifies the compute architecture (in nvcc, this is usually set by `-arch`). The `-x cuda` indicates to clang that we are handling a CUDA file (if you are handling a pure C/C++ file, we don't need this flag).

Also note that this step does not generate object files; we only instrument the code.

After this step, floating-point expressions in `compute.cu` should be instrumented. For example, if an expression originally was `y = a+b;`, it now should look like this: `y = _FPC_CHECK_(a+b, ...)`.

#### Directly Loading the Plugin

Instead of using the `clang-fpchecker` wrapper, we can directly load the plugin by adding several flags to our compilation commands.

Add the following to your CUDA compilation flags (e.g., to CXXFLAGS):

```sh
FPCHECKER_PATH      =/path/to/install
FPCHECKER_LIB       =$(FPCHECKER_PATH)/lib64/libfpchecker_plugin.so
FPCHECKER_RUNTIME   =$(FPCHECKER_PATH)/src/Runtime_plugin.h
CLANG_PLUGIN        =-Xclang -load -Xclang $(FPCHECKER_LIB) -Xclang -plugin -Xclang instrumentation_plugin
CXXFLAGS            += $(CLANG_PLUGIN) -include $(FPCHECKER_RUNTIME) -emit-llvm
```
The `$(CLANG_PLUGIN)` flag tells clang where the plugin library is and that it must load it. The `-include $(FPCHECKER_RUNTIME)` pre-includes the runtime header file. The `-emit-llvm` indicates to clang to avoid the code generation phase (we don't want to generate object files in this step; we only want to instrument the source code).

Note that since we will parse the CUDA source code with clang, we also need to add the following flags:

```sh
CUDA_OPTIONS    = --cuda-gpu-arch=sm_60 -x cuda
CXXFLAGS        += $(CUDA_OPTIONS)
```

Finally, make sure you use clang (not nvcc) as the compiler for this step:

```sh
#CXX = nvcc
CXX = clang++
```

Note that the compilation commands should not contain the `-o` flag to generate object files (we are not generating object code in this step, only transforming the source code). If the `-o` flag is added you will see this error:

```sh
clang++ $(CLANG_PLUGIN) -include $(FPCHECKER_RUNTIME) -emit-llvm .... -c file.cu -o file.o
clang-9: error: cannot specify -o when generating multiple output files
```

Like when using the `clang-fpchecker` wrapper, after this step, floating-point expressions in the code should be instrumented.

### Step 2: Compiling with nvcc

In this step, you compile the instrumented code with nvcc, as you regularly do. The only addition is that you need to pre-include the runtime header file using `-include $(FPCHECKER_RUNTIME)`; otherwise nvcc will complain about not being able to understand the `_FPC_CHECK_()` function calls.

## Requirements to the LLVM version
The primary requirement for using this version is to be able to compile your CUDA code with a recent version of the clang/LLVM compiler. Pure CUDA code or RAJA (with CUDA execution) are supported.

For more information about compiling CUDA with clang, please refer to [Compiling CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html). In particular, you should pay attention to the differences between clang/LLVM and nvcc with respect to overloading based on `__host__` and `__device__` attributes.

We have tested this version so far with these versions of clang/LLVM:
- clang 7.x
- clang 8.0

## Using the FPChecker LLVM Version
Once you are able to compile and run your CUDA application with clang, follow this to enable FPChecker:

1. Add this to your Makefile:

```sh
FPCHECKER_PATH  = /path/to/install
LLVM_PASS       = -Xclang -load -Xclang $(FPCHECKER_PATH)/lib/libcudakernels.so \
-include Runtime.h -I$(FPCHECKER_PATH)/src
CXXFLAGS += $(LLVM_PASS)
```

This will tell clang/LLVM where the FPChecker runtime is located. FPCHECKER_PATH is the where FPChecker is installed.

2. Compile your code and run it.

## Expected Output

When your applications begins to run, you should see the following (this is only visible in the LLVM version):

```sh
========================================
FPChecker (v0.1.0, Feb 21 2019)
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
The current version is not MPI aware, so every MPI process that encounters an error/warning will print a report. You should include the location of `mpi.h`; otherwise clang will not find the MPI call definitions.

## Configuration Options
Configuration options are passed via -D macros when invoking nvcc (for the clang version) or when invoking clang (for the llvm version).

In the clang version, if you are only interested in detecting the most critical exceptions, i.e., generation of NaN and Infinity numbers, use these options: `-DFPC_DISABLE_SUBNORMAL -DFPC_DISABLE_WARNINGS`.

| Option | Description | Version Available |
|--------|-------------|-----------|
| -D FPC_DISABLE_SUBNORMAL | Disable checking for subnormal numbers (underflows) | clang |
| -D FPC_DISABLE_WARNINGS | Disable warnings of small or large numbers (overflows and underflows) | clang |
| -D FPC_ERRORS_DONT_ABORT | By default FPChecker aborts the kernel that first encounters an error or warning. This option allows FPChecker to print reports without aborting. This allows you to check for errors/warnings in the entire execution of your program. | clang, llvm |
| -D FPC_DANGER_ZONE_PERCENT=x.x | Changes the size of the danger zone for warnings. By default, x.x is 0.05, and it should be a number between 0.0 and 1.0. Warning reports can be almost completely disabled by using a small danger zone, such as 0.01. | clang, llvm |

### Contact
For questions, contact Ignacio Laguna <ilaguna@llnl.gov>.

To cite FPChecker please use

```
@inproceedings{laguna2019fpchecker,
title={{FPChecker: Detecting Floating-Point Exceptions in GPU Applications}},
  author={Laguna, Ignacio},
  booktitle={2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE)},
  pages={1126--1129},
  year={2019},
  organization={IEEE}
}
```

## License

FPChecker is distributed under the terms of the Apache License (Version 2.0).

All new contributions must be made under the Apache-2.0 license.

See LICENSE and NOTICE for details.

LLNL-CODE-769426
