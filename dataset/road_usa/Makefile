GRAPH_NAME = road_usa
GRAPH = $(GRAPH_NAME).mtx
GRAPH_TAR = $(GRAPH_NAME).tar.gz
GRAPH_URL = https://www.cise.ufl.edu/research/sparse/MM/DIMACS10/$(GRAPH_TAR)

all: $(GRAPH).graph

$(GRAPH_TAR):
	wget -O $@ $(GRAPH_URL)

$(GRAPH): $(GRAPH_TAR)
	tar -xzf $(GRAPH_TAR) --strip-components=1

$(GRAPH).graph: $(GRAPH)
	../conv_mtx_to_metis.o $(GRAPH) $(GRAPH_NAME).graph