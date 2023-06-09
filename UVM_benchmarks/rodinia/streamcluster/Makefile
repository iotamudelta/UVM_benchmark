HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

HIP_FLAGS   := -I$(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib


TARGET_G = sc_gpu


# make dbg=1 tells nvcc to add debugging symbols to the binary
ifeq ($(dbg),1)
	HIP_FLAGS += -g -O0 -DPRINTINFO
else
	HIP_FLAGS += -O3 
endif

# make emu=1 compiles the HIP kernels for emulation
ifeq ($(emu),1)
	HIP_FLAGS += -deviceemu
endif

# make dp=1 compiles the HIP kernels with double-precision support
ifeq ($(dp),1)
	HIP_FLAGS += --gpu-name sm_60
endif


$(TARGET_G): streamcluster_hip_cpu.hip streamcluster_hip.hip streamcluster_header.hip
	$(HIPCC) $(HIP_FLAGS) streamcluster_hip_cpu.hip streamcluster_hip.hip streamcluster_header.hip -o $(TARGET_G)
	

clean:
	rm -f *.o *~ *.txt $(TARGET_G) *.linkinfo
	
