#!/bin/bash

export FPC_COMPILER='clang++'
export FPC_COMPILER_PARAMS=`printf "%q " "$@"`
DIR=`dirname $0`
exec "$DIR/../cpu_checking/clang_fpchecker.py"
