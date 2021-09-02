## Release 0.3.2
- Added support for printing text report
- Saves program inputs in .json trace files

## Release 0.3.1
- Fixed a number of minor bugs
- New tests added

## Release 0.3.0
- Added support to check in CPU code
- New documentation available in https://fpchecker.org/
- HTML reports provided
- Added warppers for C/C++ and MPI

## Release 0.2.1
- Added 'fpchecker' tool that intercepts all calls to nvcc from build script (e..g, make)
- The interception tool uses the Linux LD_PRELOAD trick to intercept execve system calls

## Release 0.2.0
- Added a custom front-end that instruments CUDA. This front-end doesn't require clang/LLVM.
- The new front-end is work in progress and has limitations
- Fixed several bugs
- The front-end version doesn't abort on errors by default.

## Release 0.1.2
- Added clang version. It instruments source code via clang plugin. Instrumented code can be compiled with nvcc.
- Added new flags: FPC_DISABLE_SUBNORMAL, FPC_DISABLE_WARNINGS 
- Added clang wrapper to instrument code: clang-fpchecker

## Release 0.1.0
- First stable release
- Tests use pytest

## Release 0.0.4
- When errors-dont-abort, it reports the class of error
- Various code cleaning commits
- Added some tests and cahnge how test config
- Known issue: When errors-dont-abort, we only perform FP64 checks

## Release 0.0.3
- Warning reports printed once only (in ERRORS_DONT_ABORT mode)

## Release 0.0.2
- Supports functionality to avoid aborting the program when an error/warning is found (-D FPC_ERRORS_DONT_ABORT)
- Program prints version when main() starts
- This version is not MPI-aware (all ranks print)
- Includes more tests (e.g., RAJA loop)

## Release 0.0.1
- First complete version
- It supports error and warning reports
- All reports (errors and warnings) abort after detected and printed



