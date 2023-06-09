HIP_DIR     := /opt/rocm
HIPCC       := $(HIP_DIR)/bin/hipcc
LOCAL_CC = gcc -g -O3 -Wall

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_hip.hip
	$(HIPCC) nn_hip.hip
	$(HIPCC) -o nn nn_hip.hip

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
