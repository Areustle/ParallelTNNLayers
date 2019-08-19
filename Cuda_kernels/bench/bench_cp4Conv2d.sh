#! /bin/bash

RUNNAME=$1
deviceNum=$2
# THISDATE=$(date +%F_%H_%M)

declare -a Benchmark_Executable=(
 "BatchSize"
 "ChannelDepth"
 "ImageSize"
 "FilterCount"
 "FilterSize"
)

for prob in "${Benchmark_Executable[@]}"
do
  $(./_build/Cuda_kernels/BenchCP4${prob} \
    Cuda_kernels/results/CP4Conv2dForward_results_${RUNNAME}_${prob}.txt \
    ${deviceNum})
  echo "Done ($prob)"
done
