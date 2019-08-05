#! /bin/bash

declare -a input_filter_sizes=(
    #N C H W pad fK fH fW fRank

    #Scale rank
    "1 16 32 32 1 16 3 3 1"
    "1 16 32 32 1 16 3 3 2"
    "1 16 32 32 1 16 3 3 4"
    "1 16 32 32 1 16 3 3 8"
    "1 16 32 32 1 16 3 3 16"
    "1 16 32 32 1 16 3 3 32"
    # scale RGB images
    "1 3 32 32 1 4 3 3 1"
    "1 3 64 64 1 4 3 3 1"
    "1 3 128 128 1 4 3 3 1"
    "1 3 256 256 1 4 3 3 1"
    "1 3 512 512 1 4 3 3 1"
    "1 3 1024 1024 1 4 3 3 1"
    "1 3 2048 2048 1 4 3 3 1"
    "1 3 4096 4096 1 4 3 3 1"
)

for prob in "${input_filter_sizes[@]}"
do
  RESULT=$(/opt/cuda/NsightCompute-2019.3/nv-nsight-cu-cli \
            --metrics gpu__time_duration.avg \
            --csv \
            --units base \
            ./_build/Cuda_kernels/CP4Conv2dForward \
            $prob \
                | tail -1 \
                | rev \
                | cut -d '"' -f 1-2 \
                | cut -c 2- \
                | sed 's/,//g' \
                | rev
  )
  echo "($prob),${RESULT}"
done
