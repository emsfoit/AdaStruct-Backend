from tools.hgt.pyHGT.datax import HGTGraph
import networkx as nx
import traceback

def build_hgt_graph(graph_setting, data, output_graph_file, logger):
    try:
        OAG_HGT_graph = HGTGraph(graph_setting, data, logger)
        nx.write_gpickle(OAG_HGT_graph.G, output_graph_file)
        return True
    except Exception as e:
        logger(e)
        logger(traceback.format_exc())
        return False
