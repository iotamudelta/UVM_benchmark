#!/bin/bash

# set PATH and LD_LIBRARY_PATH for CUDA/OpenCL installation (may need to be adjusted)
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib

for currDir in *
do
    echo $currDir
    if [ -d $currDir ]
	then
		cd $currDir
		pwd
		make clean
		make
		cd ..
    fi
done