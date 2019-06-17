###########
# Vars specific to local development machine
CUDA_ARCH=sm_75
CUDA_INC=-I/opt/cuda/include -I/opt
CUDA_LIB=-L/opt/cuda/lib64
###########/


.PHONY : all
all : manual cudnn_full_conv2d


%: %.cu
	nvcc -o $@ $< -O3 -std=c++14 -arch=$(CUDA_ARCH) $(CUDA_LIB) -lcudnn


.DEFAULT_GOAL := all
