###########
# Vars specific to local development machine
CUDA_ARCH=sm_75
CUDA_INC=-I/opt/cuda/include -I/opt
CUDA_LIB=-L/opt/cuda/lib64
###########/

TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')


add_one_op_kernel.cu.o: add_one_op_kernel.cu
	nvcc -o $@ $< -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -c

add_one_op_kernel.so: add_one_op_kernel.cc add_one_op_kernel.cu.o
	g++  -o $@ $^ -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart

nmode_3_2_op_kernel.cu.o: nmode_3_2_op_kernel.cu
	nvcc -o $@ $< -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -c

nmode_3_2_op_kernel.so: nmode_3_2_op_kernel.cc nmode_3_2_op_kernel.cu.o
	g++  -o $@ $^ -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart


.PHONY : all
all : add_one_op_kernel.so \
	nmode_3_2_op_kernel.so


# basic_cublas_example.out \
# zero_out.so

.PHONY: clean
clean:
	rm *.out *.so *.o

.DEFAULT_GOAL := all

# basic_cublas_example.out: example.cu
# 	nvcc -O3 -std=c++14 -arch=$(CUDA_ARCH) $(CUDA_INC) $(CUDA_LIB) -lcublas -o $@ $<

# zero_out.so: zero_out.cc
# 	g++  -O3 -std=c++14 -shared -ffast-math -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -o $@ $<

# tf_shared_libs = add_one_op_kernel.so \
# 								 nmode_3_2_op_kernel.so

# tf_kernel_objs = add_one_op_kernel.cu.o \
# 								 nmode_3_2_op_kernel.cu.o

# $(tf_shared_libs): %.so: %.cc %cu.o
# 	g++  -O3 -std=c++14 -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) $(CUDA_LIB) -lcudart -o $@ $^

# $(tf_kernel_objes): %cu.o: %.cu
# 	nvcc -O3 -std=c++14 -arch=$(CUDA_ARCH) $(TF_CFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -c -o $@ $<

# .PHONY : all
# 	all: $(tf_shared_libs)

