import heapq
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

# Graph class for Prim's algorithm
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
        df = pd.read_csv(file_path, names=['source', 'target', 'rating', 'time'], dtype={'source': int, 'target': int, 'rating': int})
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
    def prim(self, trace_file):
        undirected, nodes = self.to_undirected()
        mst = []  # Initialize fresh MST list
        visited = set()
        pq = []
        start_node = next(iter(nodes)) if nodes else None
        if not start_node:
            trace = ["Prim's MST Trace\n", "No nodes in graph, returning empty MST\n"]
            with open(trace_file, "w") as f:
                f.writelines(trace)
            return mst

        visited.add(start_node)
        trace = ["Prim's MST Trace\n", f"Starting node: {start_node}\n"]

        for neighbor in undirected[start_node]:
            heapq.heappush(pq, (1, start_node, neighbor))  # Uniform weight
            trace.append(f"Pushed to PQ: edge={start_node}->{neighbor}, weight=1\n")

        while pq and len(visited) < len(nodes):
            weight, u, v = heapq.heappop(pq)
            trace.append(f"Popped from PQ: edge={u}->{v}, weight={weight}\n")
            if v in visited:
                continue
            visited.add(v)
            edge = (u, v)
            mst.append(edge)  # Explicitly append (u, v)
            trace.append(f"Added to MST: edge={u}->{v}\n")

            for neighbor in undirected[v]:
                if neighbor not in visited:
                    heapq.heappush(pq, (1, v, neighbor))
                    trace.append(f"Pushed to PQ: edge={v}->{neighbor}, weight=1\n")

        # Log MST for debugging
        trace.append("Final MST edges:\n")
        for edge in mst:
            trace.append(f"  {edge}\n")
        if len(visited) < len(nodes):
            trace.append(f"Warning: Graph may be disconnected, MST covers {len(visited)} of {len(nodes)} nodes\n")
            # Verify connectivity
            G = self.to_networkx_undirected()
            if not nx.is_connected(G):
                trace.append("Confirmed: Graph is disconnected\n")

        with open(f"mst_prim_{len(nodes)}.txt", "w") as f:
            for u, v in mst:
                f.write(f"{u} -> {v}\n")
                print(f"MST Edge (Prim, nodes={len(nodes)}): {u} -> {v}")

        with open(trace_file, "w") as f:
            f.writelines(trace)

        G = self.to_networkx_undirected()
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=50, edge_color='gray', width=0.5)
        nx.draw_networkx_edges(G, pos, edgelist=mst, edge_color='r', width=2)
        plt.title(f"Prim's Minimum Spanning Tree ({len(self.nodes)} Nodes)")
        plt.savefig(f"mst_prim_{len(nodes)}.png", dpi=300)
        plt.close()

        return mst

def save_execution_time(algorithm, node_count, time_taken):
    """Append execution time to file."""
    with open("execution_times.txt", 'a') as f:
        f.write(f"{algorithm} (nodes={node_count}): {time_taken:.4f} seconds\n")

def save_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def plot_comparison(prim_data, kruskal_data):
    """Plot execution time vs node count as an XY line plot with markers."""
    plt.figure(figsize=(10, 6))
    if prim_data:
        nodes_p = [d['Nodes'] for d in prim_data]
        times_p = [d['Time'] for d in prim_data]
        plt.plot(nodes_p, times_p, 'r-o', label='Prim', linewidth=2, markersize=8)
    if kruskal_data:
        nodes_k = [d['Nodes'] for d in kruskal_data]
        times_k = [d['Time'] for d in kruskal_data]
        plt.plot(nodes_k, times_k, 'b-o', label='Kruskal', linewidth=2, markersize=8)
    plt.title('Execution Time vs Input Size (Prim vs Kruskal)', fontsize=14, pad=10)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('execution_time_analysis_prim.png', dpi=300)
    plt.close()

def run_prim(graph, max_nodes, dataset_path, kruskal_data):
    """Execute Prim and save results."""
    edges = graph.load_data(dataset_path, max_nodes)
    mst, duration = graph.prim(f"prim_trace_{max_nodes or 'full'}.txt")  # Unpack decorator result

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
    save_json(results, f"prim_results_{max_nodes or 'full'}.json")
    save_execution_time("Prim", len(graph.nodes), duration)
    print(f"Prim (max_nodes={max_nodes or 'full'}): {duration:.6f}s")
    result = {'Nodes': len(graph.nodes), 'Edges': edges, 'Time': duration}
    plot_comparison([result], kruskal_data)
    return result

def main():
    dataset_path = "soc-sign-bitcoinotc.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found in {os.getcwd()}")
        return

    kruskal_data = json.load(open("kruskal_plot_data.json")) if os.path.exists("kruskal_plot_data.json") else []
    results = []
    for limit in [100, 500, 1000, 2500]:
        graph = Graph()
        data = run_prim(graph, limit, dataset_path, kruskal_data)
        if data:
            results.append(data)

    if results:
        save_json(results, "prim_plot_data.json")
        plot_comparison(results, kruskal_data)
        print("Results saved to JSON")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()