PROGRAMNAME = out
MAINNAME = out
SOURCE = Main.cu

LIB1 = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.cu
LIB2 = ./FileReader/Libs/CUDA_Tri_Node_Iterator/NodeIterator.cu
LIB3 = ./FileReader/Libs/CUDA_Tri_Tensor_Multi/TensorCalculation.cu
LIB4 = ./FileReader/Libs/OpenMPTriCalc/TriangleCountingCPU.cpp
LIBEDGE = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/edge_iterator_solver.cu
FILER = ./FileReader/FileReader.cu ./FileReader/command_args.cpp
COMMONMETHODS = ./FileReader/Libs/CommonMethods/common_methods.cu
PYTHON = python3 test/FILES/gemini.py 

CUDA_LIB = $(LIB1) $(LIB2) $(LIB3) $(LIB4) 
LIBS = $(CUDA_LIB)  $(FILER) $(COMMONMETHODS)

GPU_ARCH ?= sm_89

NVCC_FLAGS = -O3 -std=c++17 -arch=$(GPU_ARCH) -rdc=true -Xcompiler -fopenmp
EXP_N_THREADS = export OMP_NUM_THREADS=12
TESTD = ./outdev -i ../test/m12.graph -v -mode 2 -b
TESTM = ./out -i ../test/m12.graph -v -mode 2 -b

.PHONY: all run clean cublas run_cublas

all: $(PROGRAMNAME)

$(PROGRAMNAME): $(SOURCE) $(LIBS) $(LIBEDGE)
	nvcc $(NVCC_FLAGS) -o $(PROGRAMNAME) $(SOURCE) $(LIBS) $(LIBEDGE); $(EXP_N_THREADS)

debug:
	nvcc -g -G -O2 -std=c++17 -arch=$(GPU_ARCH) -rdc=true -Xcompiler -fopenmp -o $(PROGRAMNAME) $(SOURCE) $(LIBS)

install: 
	sudo apt-get upgrade && sudo apt-get install build-essential nvidia-common nvidia-cuda-toolkit

compare: $(PROGRAMNAME)
	$(TESTD)
	$(TESTM)
    
clean:
	rm -f $(PROGRAMNAME) $(CUBLAS_PROG)