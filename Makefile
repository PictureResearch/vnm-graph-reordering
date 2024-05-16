CUDA_TOOLKIT := $(abspath $(shell dirname $$(command -v nvcc))/..)
ifeq ($(shell uname -m), aarch64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "sbsa-linux"
endif
endif
ifeq ($(shell uname -m), x86_64)
ifeq ($(shell uname -s), Linux)
    OS_ARCH_NVRTC := "x86_64-linux"
endif
endif


NVRTC_SHARED :=  /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so
NVCUSPARSE_SHARED := /usr/local/cuda/targets/x86_64-linux/lib/libcusparse.so
INCS         := -I./libcusparse_lt-linux-x86_64-0.5.0.1-archive/include/ -I/usr/local/cuda/targets/x86_64-linux/include/
LIBS         := -lcublas -lcudart -lcusparse -ldl ${NVCUSPARSE_SHARED} ${NVRTC_SHARED} #-L./libcusparse_lt-linux-x86_64-0.5.0.1-archive/lib -lcusparseLt

CC = nvcc
FLAGS = -arch=sm_89 -O3 -std=c++17 -w -L/usr/local/cuda/targets/x86_64-linux/lib/

spmm: main.cu 
	$(CC) $(FLAGS) ${INCS} ${LIBS} main.cu -o spmm

spmmeval: main_eval.cu 
	$(CC) $(FLAGS) ${INCS} ${LIBS} main_eval.cu -o spmmeval

clean:
	rm -f spmm spmmeval