# High-Performance CUDA Neural Network Makefile

NVCC = nvcc
CFLAGS = -O2 -Xcompiler -Wall -lcublas -arch=sm_60
PROFILE_FLAGS = -pg

SRC_DIR = src
v2_SRC = $(SRC_DIR)/version2_naive_cuda.cu
v3_SRC = $(SRC_DIR)/version3_shared_memory.cu
v4_SRC = $(SRC_DIR)/version4_optimized_convolution.cu

all: build

build: v4.exe

v2.exe: $(v2_SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

v3.exe: $(v3_SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

v4.exe: $(v4_SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

run: v4.exe
	./v4.exe

profile: v4.exe
	@echo "Running with Profiler..."
	@rm -f gmon.out
	./v4.exe --profile
	@if [ -f gmon.out ]; then \
		gprof v4.exe gmon.out > profile.txt; \
		cat profile.txt; \
	else \
		echo "No gmon.out found. Make sure you compiled with profiling flags."; \
	fi

clean:
	rm -f *.exe gmon.out profile.txt
