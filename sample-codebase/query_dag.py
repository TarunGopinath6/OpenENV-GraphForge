import os
import re
import ast
import math
import argparse
import logging
import time
import networkx as nx
from collections import Counter

from sentence_transformers import SentenceTransformer
import numpy as np

from CodebaseDAGBuilder import CodebaseDAGBuilder


# -----------------------------
# LOGGING SETUP
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# MODEL
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# CONFIG
# -----------------------------

STOPWORDS = {"add", "get", "set", "make", "do", "run"}
IGNORE_TYPES = {"external_or_function"}


# -----------------------------
# UTILS
# -----------------------------

def tokenize(text):
    return re.findall(r'\w+', text.lower())


def compute_tf(text):
    return Counter(tokenize(text))


def cosine_sim(a, b):
    common = set(a.keys()) & set(b.keys())
    num = sum(a[t] * b[t] for t in common)

    denom_a = math.sqrt(sum(v * v for v in a.values()))
    denom_b = math.sqrt(sum(v * v for v in b.values()))

    if denom_a == 0 or denom_b == 0:
        return 0

    return num / (denom_a * denom_b)


# -----------------------------
# FILTERING
# -----------------------------

def is_valid_node(node, data):
    name = node.split("::")[-1].lower()

    if name in STOPWORDS:
        logger.debug(f"Filtered STOPWORD node: {node}")
        return False

    if data.get("type") in IGNORE_TYPES:
        logger.debug(f"Filtered external node: {node}")
        return False

    if "test" in node.lower():
        logger.debug(f"Filtered test node: {node}")
        return False

    return True


# -----------------------------
# EMBEDDINGS
# -----------------------------

def build_node_embeddings(graph):
    logger.info("Building node embeddings...")
    start = time.time()

    node_texts = {}

    for node, data in graph.nodes(data=True):
        if not is_valid_node(node, data):
            continue

        text = node

        if data.get("code"):
            text += " " + data["code"][:300]

        node_texts[node] = text
        logger.debug(f"Prepared embedding text for node: {node}")

    node_list = list(node_texts.keys())
    embeddings = model.encode(list(node_texts.values()), show_progress_bar=True)

    logger.info(f"Embeddings created for {len(node_list)} nodes in {time.time() - start:.2f}s")
    return node_list, embeddings


def embed_query(query):
    logger.info("Embedding query...")
    vec = model.encode([query])[0]
    logger.debug(f"Query embedding dimension: {len(vec)}")
    return vec


def semantic_search(query_vec, node_list, node_embeddings):
    logger.info("Running semantic similarity search...")

    scores = {}

    for i, node_vec in enumerate(node_embeddings):
        sim = np.dot(query_vec, node_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(node_vec)
        )
        scores[node_list[i]] = sim

        if sim > 0.4:
            logger.debug(f"High semantic match: {node_list[i]} -> {sim:.3f}")

    return scores


# -----------------------------
# SIGNALS
# -----------------------------

def name_similarity(node, query):
    node_name = node.split("::")[-1].lower()
    query_tokens = tokenize(query)
    return sum(1 for t in query_tokens if t in node_name)


def structural_score(graph, node, query):
    neighbors = list(graph.successors(node)) + list(graph.predecessors(node))
    query_tokens = tokenize(query)

    score = 0
    for nbr in neighbors:
        for t in query_tokens:
            if t in nbr.lower():
                score += 1

    return score


# -----------------------------
# RANKING
# -----------------------------

def rank_nodes(graph, query, top_k=5):
    logger.info("Ranking nodes...")

    node_list, node_embeddings = build_node_embeddings(graph)
    query_vec = embed_query(query)

    semantic_scores = semantic_search(query_vec, node_list, node_embeddings)
    centrality = nx.degree_centrality(graph)

    ranked = []

    for node in node_list:
        data = graph.nodes[node]

        if not is_valid_node(node, data):
            continue

        semantic = semantic_scores.get(node, 0)
        name_sim = name_similarity(node, query)
        struct = structural_score(graph, node, query)
        importance = centrality.get(node, 0)

        score = (
            0.5 * semantic +
            0.3 * name_sim +
            0.2 * struct +
            0.2 * importance
        )

        logger.debug(
            f"[RANK] {node} | sem={semantic:.3f}, name={name_sim}, "
            f"struct={struct}, imp={importance:.3f}, FINAL={score:.3f}"
        )

        if semantic < 0.1 and name_sim == 0:
            continue

        ranked.append((node, score, semantic, name_sim, struct, importance))

    ranked.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Ranking complete. Top {top_k} nodes selected.")
    return ranked[:top_k]


# -----------------------------
# CAPSULE
# -----------------------------

def extract_signature(code):
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                return f"{node.name}(...):"
    except:
        pass
    return code[:80]


def build_capsule(graph, ranked_nodes):
    logger.info("Building capsule...")
    start = time.time()

    capsule = {"pivots": [], "context": []}
    seen = set()

    for node, score, sem, name_sim, struct, imp in ranked_nodes:
        logger.debug(f"Adding pivot node: {node}")

        data = graph.nodes[node]

        capsule["pivots"].append({
            "node": node,
            "score": score,
            "semantic": sem,
            "name_match": name_sim,
            "code": data.get("code", "")
        })

        seen.add(node)

        neighbors = list(graph.successors(node)) + list(graph.predecessors(node))

        for nbr in neighbors:
            if nbr in seen:
                continue

            nbr_data = graph.nodes[nbr]

            if nbr_data.get("type") in IGNORE_TYPES:
                continue

            if "test" in nbr.lower():
                continue

            logger.debug(f"Adding context node: {nbr}")

            capsule["context"].append({
                "node": nbr,
                "signature": extract_signature(nbr_data.get("code", "")),
                "type": nbr_data.get("type", "")
            })

            seen.add(nbr)

    logger.info(f"Capsule built in {time.time() - start:.2f}s")
    logger.info(f"Pivots: {len(capsule['pivots'])}, Context: {len(capsule['context'])}")

    return capsule


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to codebase")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Building DAG from codebase...")
    start = time.time()

    builder = CodebaseDAGBuilder(args.path)
    graph = builder.build()

    logger.info(f"DAG built: {len(graph.nodes)} nodes, {len(graph.edges)} edges in {time.time() - start:.2f}s")

    logger.info(f"Query: {args.query}")

    ranked = rank_nodes(graph, args.query, top_k=5)

    print("\n[✓] Top Nodes:")
    for node, score, sem, name_sim, struct, imp in ranked:
        print(
            f"{node} | score={score:.4f} "
            f"(sem={sem:.3f}, name={name_sim}, struct={struct}, imp={imp:.3f})"
        )

    capsule = build_capsule(graph, ranked)

    print("\n========== CAPSULE ==========\n")

    print("---- PIVOTS ----")
    for p in capsule["pivots"]:
        print(f"\nNode: {p['node']}")
        print(f"Score: {p['score']:.4f} | Semantic: {p['semantic']:.3f}")
        print(p["code"][:500])
        print("\n------------------")

    print("\n---- CONTEXT ----")
    for c in capsule["context"]:
        print(f"{c['node']} -> {c['signature']} [{c['type']}]")

    print("\n============================\n")


if __name__ == "__main__":
    main()