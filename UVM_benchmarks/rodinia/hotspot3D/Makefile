HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

INCLUDE     := $(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib

SRC = 3D.hip

EXE = 3D 

OUTPUT = *.out

FLAGS = -g #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(HIPCC) $(FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

enum: $(SRC)
	$(HIPCC) $(FLAGS) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

debug: $(SRC)
	$(HIPCC) $(FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

debugenum: $(SRC)
	$(HIPCC) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
