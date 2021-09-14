#!/bin/bash

export FPC_COMPILER='mpicxx'
export FPC_COMPILER_PARAMS=`printf "%q " "$@"`
DIR=`dirname $0`
exec "$DIR/../cpu_checking/mpicc_fpchecker.py"
