#!/bin/bash

CLANG_PATH=`which clang`
OMP_LIB_PATH=`dirname $CLANG_PATH`
echo $OMP_LIB_PATH/../lib
