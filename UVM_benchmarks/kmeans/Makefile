HIPCC  := /opt/rocm/bin/hipcc

all: kmeans_hip kmeans_standard 

kmeans_hip:
	$(HIPCC) -std=c++11 kmeans_hip.hip -o kmeans_hip

kmeans_standard:
	$(HIPCC) -std=c++11 standard_kmeans.hip -o kmeans_standard

clean:
	rm -f kmeans_hip kmeans_standard
