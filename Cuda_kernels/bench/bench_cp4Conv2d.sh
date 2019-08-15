#! /bin/bash

ProfBin=$1
GPUName=$2
deviceNum=$3
THISDATE=$(date +%F_%H:%M)

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
    ${ProfBin} \
    ./_build/Cuda_kernels/CP4Conv2dForward \
    Cuda_kernels/bench/tensors_${prob}.txt \
    Cuda_kernels/results/CP4Conv2dForward_results_${GPUName}_${THISDATE}_${prob}.txt \
    ${deviceNum})
  echo "Done ($prob)"
done
