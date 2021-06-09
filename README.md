# FPChecker

[![Build Status](https://travis-ci.org/LLNL/FPChecker.svg?branch=master)](https://travis-ci.org/LLNL/FPChecker)

**FPChecker** (Floating-Point Checker) is a framework to detect floating-point exceptional computations in CUDA. It is designed as a Clang/LLVM extension that instruments CUDA code to catch the result of floating-point exceptions (e.g., NaN and INF) at runtime.

## Detectable Errors and Warnings
FPChecker detects floating-point computations that produce:
- Overflows: +INF and -INF values
- Underflows: subnormal (or denormalized) values
- NANs:  e.g., from 0.0/0.0

When at least one of the threads in a CUDA grid produces any of the above cases, an error report is generated.

FPChecker can also generate **warning reports** for computations that are close to becoming overflows or underflows, i.e., `x%` from the limits of normal values, where `x` is configurable.

# Getting Started

## How to Use FPChecker

FPChecker instruments the CUDA application's code. This instrumentation can be executed in one of three ways:
- **FPChecker front-end version**: this version uses a basic front-end that instruments arithmetic operations (e.g., `x[i] = a + b ...;`) and it has no dependencies on clang/LLVM. While this version is a work in progress (see the [limtations](limitations.md)), it can instrument 99% of HPC codes and catch most errors.

- **Clang front-end version**: this version instruments the application's source code using a clang plugin. After transformations are performed, the code can be compiled with nvcc.

- **LLVM middle-end version**: this version performs instrumentation in the LLVM compiler intermediate representation (IR). It requires the CUDA application to be fully compiled with clang/LLVM.

## Building
You can build using `cmake`:
```sh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ../
make && make install
```

`cmake` will attempt to search for `clang++` and `llvm-config` in your environment. Make sure these commands are visible.

## Using the FPChecker Front-end Version

To use this version, make sure the `bin` directory is available in your `PATH` variable. In LLNL systems, run `module load fpchecker`. This version requires the use of c++11 or later, so `-std=c++11` should be added to compilation flags.

Replace `nvcc` in your build system with `nvcc-fpc` (which acts as a wrapper for nvcc). For cmake, you can use `cmake -DCMAKE_CUDA_COMPILER=nvcc-fpc`. To instrument the code at build time, the `FPC_INSTRUMENT` environment variable must be set; this can be set when running `make`:

```sh
$ FPC_INSTRUMENT=1 make -j
```
If `FPC_INSTRUMENT` is not set, the application will be compiled without instrumentation.

As an alternative, we also provide an interception tool called `fpchecker` that automatically intercepts all nvcc calls from the build script and replaces them with `nvcc-fpc`. To use this tool, simply run `fpchecker` and pass the build script (and its parameters):

```sh
$ fpchecker make -j
```

The  `fpchecker` works in Linux only; it uses the LD_PRELOAD trick to intercept all calls to nvcc (via intercepting execve()).

### Report of Instrumented Files

After the code is built, we can see a report of the processed files (which can contained instrumentation) and failed compilations:

```sh
$ fpc-report
===== FPChcecker Report =====
Processed files: 198
Failed: 0
```

If a compilation command failed, run `fpc-report -f` to see the exact failed command. To remove the compilation traces, run   `fpc-report -r`.

### Blacklisting Files

Files and lines of code can be blacklisted so that they don't get instrumented. To do that, create a configuration file named `fpchecker.ini` in the build directory, or add an environment variable `FPC_CONF` with the path of the configuration file. The configuration file format is the following:

```sh
; The following lines are not instrumented
[omit]
omit_lines = file1.cu:10-20, compute.cu:30-35, compute.cu:42-46
```

### Behavior on Errors

This version doesn't abort by default when an error is found, i.e., it will print all errors found. To abort on errors, use `-DFPC_ERRORS_ABORT`. This version also disables warnings by default. To enable warnings, use `-DFPC_ENABLE_WARNINGS`.


## Using the Clang Front-end Version

Using this version requires following two steps: (1) instrumenting the source code (with a clang plugin) and (2) compiling the code with nvcc.

### Step 1: Instrumenting the source code

This step can be executed either using the `clang-fpchecker` wrapper script or by directly loading the plugin (the `clang-fpchecker` wrapper automatically calls the required options to load the plugin). We explain both methods as follows.

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

## Requirements for the LLVM version
The primary requirement for using this version is to be able to compile your CUDA code with a recent version of the clang/LLVM compiler. Pure CUDA code or RAJA (with CUDA execution) are supported.

For more information about compiling CUDA with clang, please refer to [Compiling CUDA with clang](https://llvm.org/docs/CompileCudaWithLLVM.html). In particular, you should pay attention to the differences between clang/LLVM and nvcc with respect to overloading based on `__host__` and `__device__` attributes.

We have tested this version so far with these versions of clang/LLVM:
- clang 7.x
- clang 8.0

## Using the LLVM Version
Once you can to compile and run your CUDA application with clang, follow this to enable FPChecker:

1. Add this to your Makefile:

```sh
FPCHECKER_PATH  = /path/to/install
LLVM_PASS       = -Xclang -load -Xclang $(FPCHECKER_PATH)/lib/libcudakernels.so \
-include Runtime.h -I$(FPCHECKER_PATH)/src
CXXFLAGS += $(LLVM_PASS)
```

This will tell clang/LLVM where the FPChecker runtime is located. FPCHECKER_PATH is where FPChecker is installed.

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
