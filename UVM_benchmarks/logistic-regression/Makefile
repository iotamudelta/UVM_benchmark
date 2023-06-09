

################################ Macros #################################

SHELL       := /bin/sh
CC          := g++
# Enable debug options
# CFLAGS = -g -Wall -std=c++11
# Enable best optimization options
CFLAGS      := -Ofast -march=native -mtune=native -std=c++11
OBJECTS     := Helper.o ArffImporter.o

# HIP flags
HIPCC       := /opt/rocm/bin/hipcc
HIPCCCFLAGS := -std=c++11
HIPCCLFLAGS := ${HIPCCCFLAGS} -lhipblas

################################ Compile ################################

gpu_exec: ${OBJECTS} LogisticRegression_hipBLAS.hip
	$(HIPCC) ${HIPCCLFLAGS} -o $@ ${OBJECTS} LogisticRegression_hipBLAS.hip

Helper.o: Helper.c Helper.h BasicDataStructures.h
	$(CC) ${CFLAGS} -c Helper.c

ArffImporter.o: ArffImporter.hip ArffImporter.h BasicDataStructures.h Helper.h
	$(HIPCC) ${HIPCCCFLAGS} -c ArffImporter.hip

# exec: ${OBJECTS} LogisticRegression_Serial.c
# 	$(CC) ${CFLAGS} -o $@ ${OBJECTS} LogisticRegression_Serial.c

# Helper.o: Helper.c Helper.h BasicDataStructures.h
# 	$(CC) ${CFLAGS} -c Helper.c

# ArffImporter.o: ArffImporter.cpp ArffImporter.h BasicDataStructures.h Helper.h
# 	$(CC) ${CFLAGS} -c ArffImporter.cpp

################################# Clean #################################

clean:
	-rm -f *.o *.h.gch *exec*
