import os
import pickle
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from CodebaseDAGBuilder import CodebaseDAGBuilder


def get_repo_name(path):
    return os.path.basename(os.path.abspath(path))


def ensure_dirs():
    os.makedirs("./AST", exist_ok=True)
    os.makedirs("./AST/images", exist_ok=True)


def save_graph(graph, repo_name):
    graph_path = f"./AST/{repo_name.split('/')[-1]}.pkl"
    
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    print(f"[✓] DAG saved to {graph_path}")


def save_graph_image(graph, repo_name):
    if len(graph.nodes) > 300:
        print(f"[!] Graph too large to visualize ({len(graph.nodes)} nodes), skipping image")
        return
    image_path = f"./AST/images/{repo_name}.png"

    plt.figure(figsize=(12, 10))

    # Layout (spring is best for general graphs)
    pos = nx.spring_layout(graph, k=0.5)

    # Draw nodes + edges
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        font_size=6
    )

    plt.title(f"DAG Visualization: {repo_name}")
    plt.savefig(image_path, dpi=300)
    plt.close()

    print(f"[✓] Graph image saved to {image_path}")


def main():
    parser = argparse.ArgumentParser(description="Build AST-based DAG from codebase")
    parser.add_argument("path", help="Path to the codebase folder")

    args = parser.parse_args()

    codebase_path = args.path

    if not os.path.exists(codebase_path):
        print("[✗] Invalid path")
        return

    repo_name = get_repo_name(codebase_path)

    print(f"[→] Processing repo: {repo_name}")

    ensure_dirs()

    builder = CodebaseDAGBuilder(codebase_path)
    graph = builder.build()

    print(f"[✓] Nodes: {len(graph.nodes)}")
    print(f"[✓] Edges: {len(graph.edges)}")

    save_graph(graph, repo_name)
    save_graph_image(graph, repo_name)


if __name__ == "__main__":
    main()