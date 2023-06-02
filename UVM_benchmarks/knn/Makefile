HIPCCFLAGS =  -std=c++11 -lamdhip64 -O0
HIPCC  := /opt/rocm/bin/hipcc

knn: knn_hip.hip 
	$(HIPCC) ${HIPCCFLAGS} -o $@ $<

clean:
	rm -f knn