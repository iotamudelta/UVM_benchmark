HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

INCLUDE     := $(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib

SRC = gaussian.hip
EXE = gaussian

release: $(SRC)
	$(HIPCC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

clean:
	rm gaussian
