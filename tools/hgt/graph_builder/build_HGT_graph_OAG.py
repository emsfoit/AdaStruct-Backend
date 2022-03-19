from tools.hgt.pyHGT.datax import HGTGraph
import networkx as nx

def build_hgt_graph(graph_setting, data, output_graph_file):
    try:
        OAG_HGT_graph = HGTGraph(graph_setting, data)
        nx.write_gpickle(OAG_HGT_graph.G, output_graph_file)
        return True
    except:
        return False
