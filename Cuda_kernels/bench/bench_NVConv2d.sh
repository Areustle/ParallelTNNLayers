#! /bin/bash

GPUName=$1
deviceNum=$2
THISDATE=$(date +%F_%H_%M)

declare -a Problem_descriptor=(
 "batch_size"
 "channel_depth"
 "image_size"
 "filter_count"
 "filter_size"
)

for prob in "${Problem_descriptor[@]}"
do
  $(./_build/Cuda_kernels/BenchNV \
    Cuda_kernels/bench/tensors_${prob}.txt \
    Cuda_kernels/results/NVConv2dForward_results_${GPUName}_${THISDATE}_${prob}.txt \
    ${deviceNum})
  echo "Done ($prob)"
done
