
GRAPH = asia.osm.graph
GRAPH_BZ2 = $(GRAPH).bz2
GRAPH_URL = https://sites.cc.gatech.edu/dimacs10/archive/data/streets/$(GRAPH_BZ2)
all: $(GRAPH)

$(GRAPH_BZ2):
	wget -O $(GRAPH_BZ2) $(GRAPH_URL)

$(GRAPH): $(GRAPH_BZ2)
	bzip2 -dk $(GRAPH_BZ2)
