###########
# Vars specific to local development machine
CUDA_ARCH=sm_75
CUDA_INC=-I/opt/cuda/include -I/opt
CUDA_LIB=-L/opt/cuda/lib64
###########/

TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

basic_cublas_example.out: example.cu.cc
	nvcc -O3 -std=c++14 -arch=$(CUDA_ARCH) $(CUDA_INC) $(CUDA_LIB) -lcublas -o $@ $<

zero_out.so: zero_out.cc
	g++  -O3 -std=c++14 -shared -ffast-math -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -o $@ $<

add_one_op_kernel.cu.o: add_one_op_kernel.cu.cc
	nvcc -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -c -o $@ $<

add_one_op_kernel.so: add_one_op_kernel.cc add_one_op_kernel.cu.o
	g++  -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart -o $@ $^

# cublas_sbg_op_kernel.cu.o: cublas_sbg_op_kernel.cu.cc
# 	nvcc -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu \
# 		-Xcompiler -fPIC -c -o $@ $<

# cublas_sbg_op_kernel.so: cublas_sbg_op_kernel.cc cublas_sbg_op_kernel.cu.o
# 	g++ -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart -o $@ $^


.PHONY : all
all : add_one_op_kernel.so \
	basic_cublas_example.out \
	zero_out.so
# cublas_sbg_op_kernel.so \
	#
#
.PHONY: clean
clean:
	rm *.out *.so *.o

.DEFAULT_GOAL := all
