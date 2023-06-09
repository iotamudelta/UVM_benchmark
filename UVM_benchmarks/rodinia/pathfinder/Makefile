HIP_DIR     := /opt/rocm

HIPCC       := $(HIP_DIR)/bin/hipcc

INCLUDE     := $(HIP_DIR)/include
HIP_LIB_DIR := $(HIP_DIR)/lib
# DEF := -DPREF

SRC = pathfinder.hip

EXE = pathfinder

release:
	$(HIPCC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) $(DEF)

enum:
	$(HIPCC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(HIP_LIB_DIR) 

debug:
	$(HIPCC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(HIP_LIB_DIR) 

debugenum:
	$(HIPCC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(HIP_LIB_DIR) 

clean:
	rm -f pathfinder
