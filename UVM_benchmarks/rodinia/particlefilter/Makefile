#makefile

HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

INCLUDE     := $(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib

all: naive float

naive: ex_particle_HIP_naive_seq.hip
	$(HIPCC) -I$(INCLUDE) -L$(HIP_LIB_DIR) -g -lm -O3 -use_fast_math ex_particle_HIP_naive_seq.hip -o particlefilter_naive
	
float: ex_particle_HIP_float_seq.hip
	$(HIPCC) -I$(INCLUDE) -L$(HIP_LIB_DIR) -g -lm -O3 -use_fast_math ex_particle_HIP_float_seq.hip -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
