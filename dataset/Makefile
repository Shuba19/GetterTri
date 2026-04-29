CXX := g++
CXXFLAGS := -O3 -std=c++17

STM  := conv_snap_to_metis.cpp 
STMX := conv_snap_to_mtx.cpp 
MXTM := conv_mtx_to_metis.cpp
MTMX := conv_metis_to_mtx.cpp

OUTSTM := conv_snap_to_metis.o
OUTSTMX := conv_snap_to_mtx.o
OUTMXTM := conv_mtx_to_metis.o
OUTMTMX := conv_metis_to_mtx.o

.PHONY: conv clean dataset clean_dataset



conv:
	$(CXX) $(CXXFLAGS) $(STM) -o $(OUTSTM)
	$(CXX) $(CXXFLAGS) $(STMX) -o $(OUTSTMX)
	$(CXX) $(CXXFLAGS) $(MXTM) -o $(OUTMXTM)
	$(CXX) $(CXXFLAGS) $(MTMX) -o $(OUTMTMX)

clean:
	rm -f $(OUTSTM) $(OUTSTMX) $(OUTMXTM) $(OUTMTMX)


dataset:
	@set -e; \
		for d in */; do \
			[ -d "$$d" ] || continue; \
			echo "==> $$d"; \
			if [ -f "$$d/Makefile" ] || [ -f "$$d/makefile" ] || [ -f "$$d/GNUmakefile" ]; then \
			$(MAKE) -C "$$d"; \
			else \
			echo " skip: no Makefile"; \
			fi; \
		done 

clean_dataset:
	@set -e; \
		for d in */; do \
			[ -d "$$d" ] || continue; \
			echo "==> $$d"; \
			if [ -f "$$d/Makefile" ] || [ -f "$$d/makefile" ] || [ -f "$$d/GNUmakefile" ]; then \
			$(MAKE) -C "$$d" clean; \
			else \
			echo " skip: no Makefile"; \
			fi; \
		done