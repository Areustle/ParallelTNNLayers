#! /bin/bash

declare -a input_filter_sizes=(
    #N C H W pad fK fH fW

    #Scale rank (pointless)
    "1 16 32 32 16 3 3"

    # scale RGB images
    "1 3 32 32 4 3 3"
    "1 3 64 64 4 3 3"
    "1 3 128 128 4 3 3"
    "1 3 256 256 4 3 3"
    "1 3 512 512 4 3 3"
    "1 3 1024 1024 4 3 3"
    "1 3 2048 2048 4 3 3"
    "1 3 4096 4096 4 3 3"
)

for prob in "${input_filter_sizes[@]}"
do
  RESULT=$(/opt/cuda/NsightCompute-2019.3/nv-nsight-cu-cli \
            --metrics gpu__time_duration.avg \
            --csv \
            --units base \
            ./_build/Cuda_kernels/NVConv2dForward \
            $prob \
                | tail -1 \
                | rev \
                | cut -d '"' -f 1-2 \
                | cut -c 2- \
                | rev
  )
  echo "($prob);${RESULT}"
done
