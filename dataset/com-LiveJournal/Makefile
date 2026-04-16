
GRAPHNAME = com-lj.ungraph
GRAPH = $(GRAPHNAME).txt
GRAPH_BZ2 = $(GRAPH).gz
GRAPH_URL = https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
all: $(GRAPH)

$(GRAPH_BZ2):
	wget -O $(GRAPH_BZ2) $(GRAPH_URL)

$(GRAPH): $(GRAPH_BZ2)
	gzip -dk $(GRAPH_BZ2)
	python3 ../conv_snap_to_metis $(GRAPH) $(GRAPHNAME).graph
