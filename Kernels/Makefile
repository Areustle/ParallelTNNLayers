###########
# Vars specific to local development machine
CUDA_ARCH=sm_75
CUDA_INC=-I/opt/cuda/include -I/opt
CUDA_LIB=-L/opt/cuda/lib64
###########/

TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

.PHONY : all
all : cp_fused_nchw.so \
	rcp_fused_nchw.so \
	cp4_conv_nchw.so \
	dense_cp.so \
	manual


%.so : %.cc %.cu.o
	g++  -o $@ $^ -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart

%.cu.o: %.cu
	nvcc -o $@ $< -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -c

manual: manual.cu
	nvcc -o $@ $< -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS)


.PHONY: clean
clean:
	rm *.so

.DEFAULT_GOAL := all
