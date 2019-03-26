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



