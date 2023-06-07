HIP_DIR     := /opt/rocm
HIP_LIB_DIR := $(HIP_DIR)/lib

# C compiler
CC          := gcc
CC_FLAGS    := -g -O2 -I$(HIP_DIR)/include -D__HIP_PLATFORM_AMD__

# HIP compiler
HIPCC       := $(HIP_DIR)/bin/hipcc
HIPCC_FLAGS := -I$(HIP_DIR)/include

# DEF := -DPREF
# 'make dbg=1' enables HIPCC debugging
ifeq ($(dbg),1)
	HIPCC_FLAGS += -g -O0
else
	HIPCC_FLAGS += -O2
endif

# 'make emu=1' compiles the HIP kernels for emulation
ifeq ($(emu),1)
	HIPCC_FLAGS += -deviceemu
endif


backprop: backprop.o facetrain.o imagenet.o backprop_hip.o 
	$(HIPCC) $(CC_FLAGS) backprop.o facetrain.o imagenet.o backprop_hip.o -o backprop -L$(HIP_LIB_DIR)  -lm 

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_hip.o: backprop_hip.hip backprop.h
	$(HIPCC) $(HIPCC_FLAGS) -c backprop_hip.hip $(DEF)

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_hip.linkinfo
