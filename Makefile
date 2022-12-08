
GPP=g++ -O2 -std=c++0x 
GCC=gcc -O2
NVCC=nvcc

oddeven-seq:
	$(GPP) -o seq lab2/oddevensort.cpp
	./seq
	rm seq

oddeven-par:
	$(NVCC) -o para lab2para/oddevengpu.cu
	./para
	rm para

oddeven-debug:
	$(NVCC) -o oddevendebug lab2para/oddevengpu.cu

device-info:
	$(NVCC) -o devinfo lab2para/device_test.cu
	./devinfo
	rm devinfo