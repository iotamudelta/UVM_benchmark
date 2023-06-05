HIPCCFLAGS :=  -std=c++11 -lamdhip64
HIPCC      := /opt/rocm/bin/hipcc

all: 
	$(HIPCC) ${HIPCCFLAGS} *.hip -o svm_uvm

run:
	./svm_uvm  5 1024 93

clean:
	rm -f svm_uvm