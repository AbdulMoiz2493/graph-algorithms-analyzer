import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os

# Utility to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

# Graph class for degree calculation
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # Undirected graph
        self.nodes.add(u)
        self.nodes.add(v)

    def to_networkx_undirected(self):
        G = nx.Graph()
        for u in self.graph:
            for v in self.graph[u]:
                G.add_edge(u, v)
        return G

    def load_data(self, file_path, min_nodes=2000, max_nodes=2500):
        """Load Bitcoin OTC dataset, limiting to 2000–2500 nodes."""
        df = pd.read_csv(file_path, names=['source', 'target', 'rating', 'time'], dtype={'source': int, 'target': int, 'rating': int})
        # Filter nodes in range [1, 2500]
        df = df[(df['source'] <= max_nodes) & (df['target'] <= max_nodes)]
        edge_count = 0
        for _, row in df.iterrows():
            src, tgt = int(row['source']), int(row['target'])
            self.add_edge(src, tgt)
            edge_count += 1
        print(f"Graph: {len(self.nodes)} nodes, {edge_count} edges")
        if len(self.nodes) < min_nodes:
            print(f"Warning: Graph has {len(self.nodes)} nodes, less than minimum {min_nodes}")
        return edge_count

    @measure_time
    def calculate_degrees(self, trace_file, visualize=True):
        degrees = {node: len(self.graph[node]) for node in self.nodes}
        trace = ["Node Degree Calculation Trace\n"]

        for node in sorted(degrees.keys()):
            trace.append(f"Node {node}: Degree = {degrees[node]}\n")

        with open("node_degrees.txt", "w") as f:
            for node, degree in degrees.items():
                f.write(f"Node {node}: Degree = {degree}\n")
                print(f"Node {node}: Degree = {degree}")

        with open(trace_file, "w") as f:
            f.writelines(trace)

        if visualize:
            G = self.to_networkx_undirected()
            pos = nx.spring_layout(G, k=0.15, iterations=20)
            plt.figure(figsize=(12, 8))
            node_sizes = [degrees.get(node, 1) * 50 for node in G.nodes()]
            nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=node_sizes, edge_color='gray', width=0.5)
            plt.title(f"Graph with {len(self.nodes)} Nodes (Node Size Proportional to Degree)")
            plt.savefig("degree_visualization.png", dpi=300)
            plt.close()

        return degrees

def save_execution_time(algorithm, node_count, time_taken):
    """Append execution time to file."""
    with open("execution_times.txt", 'a') as f:
        f.write(f"{algorithm} (nodes={node_count}): {time_taken:.4f} seconds\n")

def main():
    dataset_path = "soc-sign-bitcoinotc.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found in {os.getcwd()}")
        return

    g = Graph()
    edge_count = g.load_data(dataset_path, min_nodes=2000, max_nodes=2500)
    degrees, time_taken = g.calculate_degrees("degree_trace.txt")
    
    print(f"Degree calculation execution time: {time_taken:.6f} seconds")
    save_execution_time("Degree", len(g.nodes), time_taken)
    
    with open("degree_execution_time.txt", "w") as f:
        f.write(f"Degree Calculation: {time_taken:.6f} seconds\n")
    
    print(f"Results saved: node_degrees.txt, degree_trace.txt, degree_visualization.png")

if __name__ == "__main__":
    main()