#! /bin/bash

GPUName=$1
deviceNum=$2

declare -a Problem_descriptor=(
 "batch_size"
 "channel_depth"
 "image_size"
 "filter_count"
 "filter_size"
)

for prob in "${Problem_descriptor[@]}"
do
  $(./_build/Cuda_kernels/Tensor_Benchmark \
    ./_build/Cuda_kernels/CP4Conv2dForward \
    Cuda_kernels/bench/tensors_${prob}.txt \
    Cuda_kernels/results/CP4Conv2dForward_results_${GPUName}_$(date +'%F')_${prob}.txt \
    ${deviceNum})
  echo "Done ($prob)"
done
