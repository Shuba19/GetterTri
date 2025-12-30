PROGRAMNAME = out
MAINNAME = out
SOURCE = Main.cu
LIB1 = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.cu
LIB2 = ./FileReader/Libs/CUDA_Tri_Node_Iterator/NodeIterator.cu
LIB3 = ./FileReader/Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.cu
FILER = ./FileReader/FileReader.cu ./FileReader/command_args.cpp
COMMONMETHODS = ./FileReader/Libs/CommonMethods/common_methods.cu
PYTHON = python3 test/FILES/gemini.py 
CUDA_LIB = $(LIB1) $(LIB2) $(LIB3) $(LIB4) 
LIBS = $(CUDA_LIB)  $(FILER) $(COMMONMETHODS)

GPU_ARCH ?= sm_89

TESTD = ./outdev -i ../test/m12.graph -v -mode 2 -b
TESTM = ./out -i ../test/m12.graph -v -mode 2 -b

.PHONY: all run clean cublas run_cublas

all: $(PROGRAMNAME)


$(PROGRAMNAME): $(SOURCE) $(LIBS)
	nvcc -O3 -std=c++17 -arch=$(GPU_ARCH) -rdc=true  -o $(PROGRAMNAME) $(SOURCE) $(LIBS)

debug:
	nvcc -g -G -O2 -std=c++17 -arch=$(GPU_ARCH) -rdc=true  -o $(PROGRAMNAME) $(SOURCE)  $(LIBS)
install: 
	sudo apt-get upgrade && sudo apt-get install build-essential nvidia-common nvidia-cuda-toolkit
compare: $(PROGRAMNAME)
	 $(TESTD)
	 $(TESTM)
	
clean:
	rm -f $(PROGRAMNAME) $(CUBLAS_PROG)

