<!--
# FPChecker
-->
<img src="figures/logo_fpchecker.png" />

[![Build Status](https://travis-ci.org/LLNL/FPChecker.svg?branch=master)](https://travis-ci.org/LLNL/FPChecker)

**FPChecker** (Floating-Point Checker) is a dynamic analysis tool to detect floating-point errors in HPC applications. It is the only tool of its class that supports the most common programming languages and models in HPC, including C/C++, MPI, OpenMP, and CUDA. It is designed to be easy to use and easy to integrate into applications. The tool provides a detailed HTML report that helps users identify the exact location of floating-point errors in the software.

## Documentation

To see how to install and use FPChecker, visit: [https://fpchecker.org/](https://fpchecker.org/)

## Features

- **Easy to use:** it only requires a few changes to the application build script, such as changing the compiler (e.g., clang++) by the FPChecker compiler wrappers (e.g., clang++-fpchecker). It automatically instruments the code at build time.
- **Accurate Detection:** it accurately detects errors dynamically (when code is executed) for specific inputs; it doesn’t give alarms for unused or invalid inputs. 
- **Design for HPC:** it supports the most used programming languages and models in HPC: C/C++, MPI, OpenMP, Pthreads, and CUDA.
- **Detailed report:** it provides a detailed report that programmers can use to identify the exact location (file and line number) of floating-point errors in the software.

## Errors and Warnings

FPChecker detects the following floating-point issues:

- **Infinity +:** detected when operations produce positive infinity, for example, when 1.0 / 0.0 occurs. 
- **Infinity -:** this is the same as infinity +, except that the sign of the resulting calculation is negative.
- **NaN:** NaN (not a number) results from invalid operations, such as 0/0 or sqrt(-1).
- **Division by zero:** This occurs when a finite nonzero number is divided by zero. This typically produces either infinity or NaN.
- **Underflow (subnormal):** Underflow is detected when an operation produces a subnormal number because the result was not representable as a normal number.
- **Comparison:** This occurs when two floating-point numbers are compared for equality. Sometimes checking if two floating-point numbers are equal can lead to inaccuracies.
- **Cancellation:** cancellation occurs when two nearly equal numbers are subtracted. By default, this event is detected when at least ten decimal digits are lost due to a subtraction.
- **Latent Infinity +:** is detected when an operation produces a large normal and is close to positive infinity.
- **Latent Infinity -:** is detected when an operation produces a large normal number and is close to negative infinity.
- **Latent underflow:** is detected when an operation produces a small normal number and is close to becoming an underflow (subnormal number).

## How FPChecker Works

FPChecker is designed as an extension of the clang/LLVM compiler. When the application is compiled, an LLVM pass instruments the LLVM IR code after optimizations and inserts check code to all floating-point operations. The check code calls routines in the FPChecker runtime system, which detects several floating-point events (see above). When the code execution ends, traces are saved in the current directory. These traces are then used to build a detailed report of the location of the detected events.

### Contact
For questions, contact Ignacio Laguna <ilaguna@llnl.gov>.

To cite FPChecker please use

```
Laguna, Ignacio. "FPChecker: Detecting Floating-point Exceptions in GPU Applications."
In 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), 
pp. 1126-1129. IEEE, 2019.
```

## License

FPChecker is distributed under the terms of the Apache License (Version 2.0).

All new contributions must be made under the Apache-2.0 license.

See LICENSE and NOTICE for details.

LLNL-CODE-769426
