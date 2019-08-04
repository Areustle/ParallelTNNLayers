#! /bin/bash

RESULT=$(/opt/cuda/NsightCompute-2019.3/nv-nsight-cu-cli \
          --metrics gpu__time_duration.avg \
          --csv \
          --units base \
          ./_build/Cuda_kernels/bench_cp4conv \
              | tail -1 \
              | rev \
              | cut -d '"' -f 1-2 \
              | cut -c 2- \
              | sed 's/,//g' \
              | rev
            )

echo "${RESULT} ns"
