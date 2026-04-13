PROGRAMNAME = out
MAINNAME = out
SOURCE = Main.cu

LIB1 = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/EdgeIterator.cu
LIB2 = ./FileReader/Libs/CUDA_Tri_Node_Iterator/NodeIterator.cu
LIB3 = ./FileReader/Libs/CUDA_Tri_Tensor_Multi/
LIB4 = ./FileReader/Libs/OpenMPTriCalc/TriangleCountingCPU.cpp
LIBTENSORS = $(LIB3)TensorCalculation.cu $(LIB3)TensorUtilities.cu $(LIB3)TTC2.cu $(LIB3)TTC3.cu $(LIB3)TTC4.cu
LIBEDGE = ./FileReader/Libs/CUDA_Tri_Edge_Iterator/edge_iterator_solver.cu
FILER = ./FileReader/FileReader.cu ./FileReader/command_args.cpp
COMMONMETHODS = ./FileReader/Libs/CommonMethods/common_methods.cu
PYTHON = python3 test/FILES/gemini.py 

CUDA_LIB = $(LIB1) $(LIB2) $(LIBTENSORS)  $(LIB4)
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
    
test:
	for graph in m12.graph m11.graph m10.graph; do \
		echo "Running tests on $$graph with different modes..."; \
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"; \
		./$(PROGRAMNAME) -i dataset/METIS/$$graph  -mode 0; \
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++" ; \
		./$(PROGRAMNAME) -i dataset/METIS/$$graph  -mode 2; \
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++" ; \
		./$(PROGRAMNAME) -i dataset/METIS/$$graph  -mode 5; \
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"; \
		./$(PROGRAMNAME) -i dataset/METIS/$$graph  -mode 6; \
		echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"; \
		echo "" ; \
	done
report:

	for graph in  m11.graph m10.graph tiles.graph; do \
		for mode in 0 2 5 6; do \
			echo "Running $(PROGRAMNAME) on $$graph with mode $$mode..."; \
			ncu --set full -f -o out_mode$${mode}_$${graph} ./$(PROGRAMNAME) -i dataset/METIS/$$graph -mode $$mode -v; \
			echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"; \
		done; \
	done 
clean:
	rm -f $(PROGRAMNAME) $(CUBLAS_PROG)
bench:
	ncu --set full -f -o out_mode$(MODE)_m10 ./$(PROGRAMNAME) -i dataset/METIS/m10.graph -mode $(MODE) -v
	ncu --set full -f -o out_mode$(MODE)_m11 ./$(PROGRAMNAME) -i dataset/METIS/m11.graph -mode $(MODE) -v
	ncu --set full -f -o out_mode$(MODE)_m12 ./$(PROGRAMNAME) -i dataset/METIS/m12.graph -mode $(MODE) -v