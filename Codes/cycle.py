import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Utility to measure execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

# Graph class for Cycle Detection
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = set()

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.nodes.add(u)
        self.nodes.add(v)

    def to_networkx(self):
        G = nx.DiGraph()
        for u in self.graph:
            for v in self.graph[u]:
                G.add_edge(u, v)
        return G

    @measure_time
    def detect_cycle(self, trace_file):
        visited = set()
        rec_stack = set()
        trace = ["Cycle Detection Trace\n"]
        cycle_nodes = []

        def dfs_cycle(node, parent):
            visited.add(node)
            rec_stack.add(node)
            trace.append(f"Visiting: {node}, Rec Stack: {rec_stack}\n")
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs_cycle(neighbor, node):
                        cycle_nodes.append((node, neighbor))
                        return True
                elif neighbor in rec_stack and neighbor != parent:
                    cycle_nodes.append((node, neighbor))
                    trace.append(f"Cycle detected: {node} -> {neighbor}\n")
                    return True
            rec_stack.remove(node)
            trace.append(f"Backtracking from: {node}\n")
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs_cycle(node, None):
                    cycle = []
                    if cycle_nodes:
                        start, end = cycle_nodes[-1]
                        cycle.append(end)
                        current = start
                        while current != end and current in rec_stack:
                            cycle.append(current)
                            for next_node in self.graph[current]:
                                if next_node in rec_stack:
                                    current = next_node
                                    break
                        cycle.reverse()
                    with open("cycle_detection.txt", "w") as f:
                        if cycle:
                            f.write(f"Cycle detected: {' -> '.join(map(str, cycle))}\n")
                            print(f"Cycle detected: {' -> '.join(map(str, cycle))}")
                        else:
                            f.write("Cycle detected, but specific nodes not fully traced.\n")
                            print("Cycle detected, but specific nodes not fully traced.")
                    with open(trace_file, "w") as f:
                        f.writelines(trace)
                    G = self.to_networkx()
                    pos = nx.spring_layout(G)
                    plt.figure(figsize=(10, 8))
                    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
                    if cycle:
                        cycle_edges = [(cycle[i], cycle[i+1]) for i in range(len(cycle)-1)] + [(cycle[-1], cycle[0])]
                        nx.draw_networkx_edges(G, pos, edgelist=cycle_edges, edge_color='r', width=2)
                    plt.title("Detected Cycle")
                    plt.savefig("cycle.png")
                    plt.close()
                    return True

        with open("cycle_detection.txt", "w") as f:
            f.write("No cycle detected in the graph.\n")
        print("No cycle detected in the graph")
        with open(trace_file, "w") as f:
            f.writelines(trace)
        return False

    def analyze_runtime(self):
        # Visualization of all nodes from dataset
        G = self.to_networkx()
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 10))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=8)
        plt.title("Visualization of All Nodes from Dataset")
        plt.savefig("all_nodes_visualization.png")
        plt.close()

        # Runtime comparison for first 100-200 nodes from dataset
        configs = [
            ("Sparse Directed", True, 2),
            ("Dense Directed", True, 0.1),
        ]
        algorithms = [
            ("Cycle Detection", lambda g: g.detect_cycle("cycle_temp_trace.txt"))
        ]

        # Visualization for first 100-200 nodes
        self.graph.clear()
        self.nodes.clear()
        node_set = set()
        edge_count = 0
        with open("dataset.csv", "r") as f:
            next(f)
            for line in f:
                u, v = map(int, line.strip().split(","))
                self.add_edge(u, v)
                node_set.add(u)
                node_set.add(v)
                edge_count += 1
                if len(node_set) >= 200:
                    break
        print(f"Using {len(node_set)} nodes and {edge_count} edges from dataset.csv")

        G_100_200 = self.to_networkx()
        pos_100_200 = nx.spring_layout(G_100_200)
        plt.figure(figsize=(10, 8))
        nx.draw(G_100_200, pos_100_200, with_labels=True, node_color='lightgreen', node_size=400, font_size=10)
        plt.title(f"Visualization of First {len(node_set)} Nodes (Directed)")
        plt.savefig("visualization_100_200_nodes.png")
        plt.close()

        # Runtime comparison bar chart for first 100-200 nodes
        plt.figure(figsize=(12, 8))
        bar_width = 0.35
        index = range(len(algorithms))
        for config_idx, (config_name, directed, edge_factor) in enumerate(configs):
            self.graph.clear()
            self.nodes.clear()
            node_set = set()
            edge_count = 0
            with open("dataset.csv", "r") as f:
                next(f)
                for line in f:
                    u, v = map(int, line.strip().split(","))
                    self.add_edge(u, v)
                    node_set.add(u)
                    node_set.add(v)
                    edge_count += 1
                    if len(node_set) >= 200:
                        break
            print(f"Config: {config_name}, Nodes: {len(node_set)}, Edges: {edge_count}")

            for algo_idx, (algo_name, algo_func) in enumerate(algorithms):
                result, time_taken = algo_func(self)
                plt.bar(algo_idx + config_idx * bar_width, time_taken, bar_width, label=config_name if algo_idx == 0 else None)
        plt.xlabel("Algorithm")
        plt.ylabel("Execution Time (seconds)")
        plt.title(f"Runtime Comparison for First {len(node_set)} Nodes")
        plt.xticks([i + bar_width/2 for i in index], [algo[0] for algo in algorithms])
        plt.legend()
        plt.savefig("runtime_comparison_100_200_nodes.png")
        plt.close()

        # XY plot for input sizes from 10 to 100 nodes
        node_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        edge_factor = 2
        times = []
        print("Analyzing Cycle Detection for sparse directed graph...")
        for n in node_counts:
            self.graph.clear()
            self.nodes.clear()
            G = nx.gnp_random_graph(n, edge_factor/n, directed=True)
            edge_count = 0
            for u, v in G.edges():
                self.add_edge(u, v)
                edge_count += 1
            print(f"  Nodes: {n}, Edges: {edge_count}")

            result, time_taken = self.detect_cycle("cycle_temp_trace.txt")
            times.append(time_taken)

        plt.figure(figsize=(12, 8))
        plt.plot(node_counts, times, color='blue')
        plt.title("Graph of Execution time of Cycle Detection")
        plt.xlabel("Input Size (Number of Nodes)")
        plt.ylabel("Running Time (seconds)")
        plt.grid(True)
        plt.xlim(10, 100)
        plt.ylim(0, max(times) * 1.2 if times else 1)
        plt.savefig("runtime_cycle_detection.png")
        plt.close()
        print("Plot saved: runtime_cycle_detection.png")

def main():
    g = Graph()
    with open("dataset.csv", "r") as f:
        next(f)
        for line in f:
            u, v = map(int, line.strip().split(","))
            g.add_edge(u, v)

    has_cycle, time_taken = g.detect_cycle("cycle_trace.txt")
    print(f"Cycle detection execution time: {time_taken:.6f} seconds")
    with open("cycle_detection_execution_time.txt", "w") as f:
        f.write(f"Cycle Detection: {time_taken:.6f} seconds\n")

    print("Running runtime analysis...")
    g.analyze_runtime()
    print("Runtime analysis complete. Plots saved as all_nodes_visualization.png, visualization_100_200_nodes.png, runtime_comparison_100_200_nodes.png, and runtime_cycle_detection.png")

if __name__ == "__main__":
    main()