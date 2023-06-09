all:
	/opt/rocm/bin/hipcc -O3 ${HIPFILES} ${DEF} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe