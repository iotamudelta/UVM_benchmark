HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

INCLUDE     := $(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib

SRC = srad.hip

EXE = srad

release: $(SRC)
	$(HIPCC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

enum: $(SRC)
	$(HIPCC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

debug: $(SRC)
	$(HIPCC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

debugenum: $(SRC)
	$(HIPCC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
