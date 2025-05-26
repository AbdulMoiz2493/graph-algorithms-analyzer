import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os
import json

# Utility to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

# Graph class for Kruskal's algorithm
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # Undirected graph
        self.nodes.add(u)
        self.nodes.add(v)

    def to_undirected(self):
        undirected = defaultdict(list)
        nodes = set()
        for u in self.graph:
            for v in self.graph[u]:
                undirected[u].append(v)
                undirected[v].append(u)
                nodes.add(u)
                nodes.add(v)
        return undirected, nodes

    def to_networkx_undirected(self):
        G = nx.Graph()
        undirected, _ = self.to_undirected()
        for u in undirected:
            for v in undirected[u]:
                G.add_edge(u, v)
        return G

    def load_data(self, file_path, max_nodes=None):
        """Load Bitcoin OTC dataset, optionally limiting nodes."""
        try:
            df = pd.read_csv(file_path, names=['source', 'target', 'rating', 'time'], dtype={'source': int, 'target': int, 'rating': int})
        except FileNotFoundError:
            print(f"Error: {file_path} not found in {os.getcwd()}")
            return 0
        except pd.errors.EmptyDataError:
            print(f"Error: {file_path} is empty")
            return 0
        if max_nodes:
            df = df[(df['source'] <= max_nodes) & (df['target'] <= max_nodes)]
        edge_count = 0
        for _, row in df.iterrows():
            src, tgt = int(row['source']), int(row['target'])
            self.add_edge(src, tgt)
            edge_count += 1
        print(f"Graph: {len(self.nodes)} nodes, {edge_count} edges")
        return edge_count

    @measure_time
    def kruskal(self, trace_file):
        undirected, nodes = self.to_undirected()
        mst = []  # Fresh MST list
        trace = ["Kruskal's MST Trace\n"]
        if not nodes:
            trace.append("No nodes in graph, returning empty MST\n")
            with open(trace_file, "w") as f:
                f.writelines(trace)
            return mst

        edges = []
        seen = set()
        for u in undirected:
            for v in undirected[u]:
                edge = tuple(sorted([u, v]))
                if edge not in seen:
                    edges.append((1, u, v))  # Uniform weight
                    seen.add(edge)
        edges.sort()  # Sort by weight (all 1, so sorts by nodes)
        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                elif rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1

        for weight, u, v in edges:
            trace.append(f"Considering edge: {u}->{v}, weight={weight}\n")
            if find(u) != find(v):
                union(u, v)
                edge = (u, v)
                mst.append(edge)
                trace.append(f"Added to MST: edge={u}->{v}\n")

        # Log MST for debugging
        trace.append("Final MST edges:\n")
        for edge in mst:
            trace.append(f"  {edge}\n")
        # Check connectivity
        G = self.to_networkx_undirected()
        if not nx.is_connected(G):
            trace.append(f"Warning: Graph is disconnected, MST is a forest covering {len(mst) + 1} nodes\n")
        else:
            trace.append(f"Graph is connected, MST covers {len(mst) + 1} nodes\n")

        try:
            with open(f"mst_kruskal_{len(nodes)}.txt", "w") as f:
                for u, v in mst:
                    f.write(f"{u} -> {v}\n")
                    print(f"MST Edge (Kruskal, nodes={len(nodes)}): {u} -> {v}")
        except IOError as e:
            print(f"Error writing to mst_kruskal_{len(nodes)}.txt: {e}")

        try:
            with open(trace_file, "w") as f:
                f.writelines(trace)
        except IOError as e:
            print(f"Error writing to {trace_file}: {e}")

        try:
            pos = nx.spring_layout(G, k=0.15, iterations=20)
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=50, edge_color='gray', width=0.5)
            nx.draw_networkx_edges(G, pos, edgelist=mst, edge_color='r', width=2)
            plt.title(f"Kruskal's Minimum Spanning Tree ({len(self.nodes)} Nodes)")
            plt.savefig(f"mst_kruskal_{len(nodes)}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error generating mst_kruskal_{len(nodes)}.png: {e}")

        return mst

def save_execution_time(algorithm, node_count, time_taken):
    """Append execution time to file."""
    try:
        with open("execution_times.txt", 'a') as f:
            f.write(f"{algorithm} (nodes={node_count}): {time_taken:.4f} seconds\n")
        with open(f"{algorithm.lower()}_execution_time.txt", "w") as f:
            f.write(f"{algorithm}: {time_taken:.6f} seconds\n")
    except IOError as e:
        print(f"Error writing execution time: {e}")

def save_json(data, filename):
    """Save data to JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error writing to {filename}: {e}")

def plot_comparison(kruskal_data, prim_data):
    """Plot execution time vs node count as an XY line plot with markers."""
    try:
        plt.figure(figsize=(10, 6))
        if kruskal_data:
            nodes_k = [d['Nodes'] for d in kruskal_data]
            times_k = [d['Time'] for d in kruskal_data]
            plt.plot(nodes_k, times_k, 'b-o', label='Kruskal', linewidth=2, markersize=8)
        if prim_data:
            nodes_p = [d['Nodes'] for d in prim_data]
            times_p = [d['Time'] for d in prim_data]
            plt.plot(nodes_p, times_p, 'r-o', label='Prim', linewidth=2, markersize=8)
        plt.title('Execution Time vs Input Size (Kruskal vs Prim)', fontsize=14, pad=10)
        plt.xlabel('Number of Nodes', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('execution_time_analysis_kruskal.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating execution_time_analysis_kruskal.png: {e}")

def run_kruskal(graph, max_nodes, dataset_path, prim_data):
    """Execute Kruskal and save results."""
    edges = graph.load_data(dataset_path, max_nodes)
    if edges == 0:
        print(f"No edges loaded for max_nodes={max_nodes or 'full'}, skipping")
        return None
    mst, duration = graph.kruskal(f"kruskal_trace_{max_nodes or 'full'}.txt")  # Unpack decorator result

    # Validate MST edges
    valid_mst = []
    for edge in mst:
        if not isinstance(edge, tuple) or len(edge) != 2:
            print(f"Warning: Invalid MST edge {edge}, expected (u, v) tuple")
            continue
        valid_mst.append(edge)

    results = {
        'max_nodes': max_nodes or 'full',
        'node_count': len(graph.nodes),
        'edge_count': edges,
        'time': duration,
        'mst_edges': [(u, v) for u, v in valid_mst]
    }
    save_json(results, f"kruskal_results_{max_nodes or 'full'}.json")
    save_execution_time("Kruskal", len(graph.nodes), duration)
    print(f"Kruskal (max_nodes={max_nodes or 'full'}): {duration:.6f}s")
    result = {'Nodes': len(graph.nodes), 'Edges': edges, 'Time': duration}
    plot_comparison([result], prim_data)
    return result

def main():
    dataset_path = "soc-sign-bitcoinotc.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found in {os.getcwd()}")
        return

    try:
        prim_data = json.load(open("prim_plot_data.json")) if os.path.exists("prim_plot_data.json") else []
    except json.JSONDecodeError:
        print("Warning: prim_plot_data.json is corrupted, using empty data")
        prim_data = []

    results = []
    for limit in [100, 500, 1000, 2500]:
        graph = Graph()
        data = run_kruskal(graph, limit, dataset_path, prim_data)
        if data:
            results.append(data)

    if results:
        save_json(results, "kruskal_plot_data.json")
        plot_comparison(results, prim_data)
        print("Results saved to JSON")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()