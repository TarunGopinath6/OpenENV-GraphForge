import pickle
import networkx as nx

def load_graph(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def extract_ego_subgraph(graph, central_node, radius=1):
    matching = [n for n in graph.nodes if n.endswith(f"::{central_node}")]
    
    if not matching:
        print(f"[✗] No node found matching '::{central_node}'")
        return None
    
    if len(matching) > 1:
        print(f"[!] Multiple matches found:")
        for i, n in enumerate(matching):
            print(f"  [{i}] {n}")
        choice = int(input("Select node index: "))
        seed = matching[choice]
    else:
        seed = matching[0]
    
    print(f"[✓] Seed node: {seed}")
    
    subgraph = nx.ego_graph(graph, seed, radius=radius, undirected=True)
    
    print(f"[✓] Subgraph — Nodes: {len(subgraph.nodes)}, Edges: {len(subgraph.edges)}")
    return subgraph