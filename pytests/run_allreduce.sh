#!/bin/bash

set -xe

cd $(dirname $0)
export NCCL_PROFILER_PLUGIN=../build/libnccltrace.so
pushd ../build
make -j $(nproc) nccltrace
popd
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE
export NCCL_TRACER_DUMP_FILE_NAME=$PWD/all_reduce_log
rm -rf all_reduce_log.rank_*
torchrun --nproc_per_node=2 --standalone all_reduce.py