ROCM_INSTALL_PATH ?= /opt/rocm/
GCC_VER = -5

CXX := /usr/bin/g++
CC := /usr/bin/gcc

HIPCC  := $(ROCM_INSTALL_PATH)bin/hipcc
LINK := $(HIPCC)

INCLUDES := -I. -I$(ROCM_INSTALL_PATH)include -D__HIP_PLATFORM_AMD__
# Libraries
LIB_ROCM := -L/opt/rocm/lib -lamdhip64

# Options
HIPCCOPTIONS := -std=c++11  -Wno-deprecated-gpu-targets
CXXOPTIONS = -std=c++11 -O2 -D__HIP_PLATFORM_AMD__

COMMONFLAGS += $(INCLUDES) 
HIPCCFLAGS += $(HIPCCOPTIONS)  $(COMMONFLAGS) 
CXXFLAGS += $(COMMONFLAGS) $(CXXOPTIONS)
CFLAGS += $(COMMONFLAGS)
TARGET = main
OBJS =  graph.c.o bfsCPU.c.o main.hip.o bfsHIP.hip.o
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_ROCM)



$(TARGET):  $(OBJS) 
	$(LINKLINE)

%.hip.o: %.hip
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

%.c.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(TARGET) *.o 
