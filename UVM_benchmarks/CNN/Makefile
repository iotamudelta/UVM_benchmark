HIPCC  := /opt/rocm/bin/hipcc

all:
	$(HIPCC) -lamdhip64 -lhipblas main.hip layer.hip -o CNN

run:
	./CNN
clean:
	rm -f CNN
