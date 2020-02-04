#!/bin/bash

NVCC_PATH=$(which nvcc)
CUDA_PATH=$(dirname ${NVCC_PATH})
CUDA_PATH=$(dirname ${CUDA_PATH})
#echo "${CUDA_PATH}"

CUDA_LIB_PATH="${CUDA_PATH}/lib"
CUDA_LIB64_PATH="${CUDA_PATH}/lib64"

if [ -d ${CUDA_LIB_PATH} ]; then
  lib_path="${CUDA_LIB_PATH}/libcudart.so"
  if [ -f "$lib_path" ]; then
    echo "${CUDA_LIB_PATH}"
  fi
elif [ -d ${CUDA_LIB64_PATH} ]; then
  lib_path="${CUDA_LIB64_PATH}/libcudart.so"
  if [ -f "$lib_path" ]; then
    echo "${CUDA_LIB64_PATH}"
  fi
fi
