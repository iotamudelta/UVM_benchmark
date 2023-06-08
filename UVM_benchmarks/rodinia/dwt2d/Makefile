#Configure
HIP_DIR    := /opt/rocm
EXECUTABLE := dwt2d

# Debug options
dbg=0
cudagdb=0 #compile for use with cuda-gdb, note that 'dbg' must be 1 as well
performancetest=1

# HIPCC Options
# HIPCCFLAGS += -arch sm_20

# Files
CFILES   := 
CXXFILES := 
HIPFILES := main.hip dwt.hip components.hip dwt_cuda/fdwt53.hip dwt_cuda/fdwt97.hip dwt_cuda/common.hip dwt_cuda/rdwt97.hip dwt_cuda/rdwt53.hip

# Includes
INCLUDES := -I. -I$(HIP_DIR)/include


# Common flags
COMMONFLAGS += $(INCLUDES) 
HIPCCFLAGS  += $(COMMONFLAGS)
CXXFLAGS    += $(COMMONFLAGS)
CFLAGS      += $(COMMONFLAGS) -std=c99 
LDFLAGS     += -L$(HIP_DIR)/lib64

# Warning flags (from cuda common.mk)
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CFLAGS += $(CWARN_FLAGS)
CXXFLAGS += $(CXXWARN_FLAGS)

# Debug/release flags
ifeq ($(dbg),1)
    COMMONFLAGS += -g 
    HIPCCFLAGS  += -D_DEBUG
    CXXFLAGS    += -D_DEBUG
    CFLAGS      += -D_DEBUG

    ifeq ($(cudagdb),1)
        HIPCCFLAGS += -G
    endif
else 
    COMMONFLAGS += -O2 
    HIPCCFLAGS  += -fno-strict-aliasing
    CXXFLAGS    += -fno-strict-aliasing
    CFLAGS      += -fno-strict-aliasing
endif

#ifeq ($(performancetest),1)
#    COMMONFLAGS += -DGPU_DWT_TESTING
#endif

#ifdef OUTPUT
#	override OUTPUT = -DOUTPUT
#endif

OUTPUT = -DOUTPUT

# Compilers
CXX   := g++
CC    := gcc
HIPCC := $(HIP_DIR)/bin/hipcc
LINK  := $(HIPCC)

# Generate object files list
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
HIPOBJS=$(HIPFILES:.hip=.hip.o)

.SUFFIXES: .c.o .cpp.o .hip.o .hip

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.hip.o: %.hip
	$(HIPCC) $(OUTPUT) $(HIPCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(COBJS) $(CXXOBJS) $(HIPOBJS) 
	$(LINK) -o $(EXECUTABLE) $(COBJS) $(CXXOBJS) $(HIPOBJS) $(LDFLAGS)

clean:
	rm -f $(COBJS) $(CXXOBJS) $(HIPOBJS) $(EXECUTABLE)
	rm *.bmp.dwt.*		