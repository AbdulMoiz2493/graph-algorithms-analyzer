import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import os

class Graph:
    """Directed, weighted graph with edge list for Bellman-Ford."""
    def __init__(self):
        self.adj = defaultdict(list)  # {node: [(neighbor, weight), ...]}
        self.nodes = set()
        self.all_edges = []  # (src, dst, weight)

    def insert_edge(self, src, dst, wt):
        """Add edge to adjacency list and edge list."""
        self.adj[src].append((dst, wt))
        self.all_edges.append((src, dst, wt))
        self.nodes.add(src)
        self.nodes.add(dst)

    def neighbors(self, node):
        """Get (neighbor, weight) pairs."""
        return self.adj.get(node, [])

    def degree_out(self, node):
        """Count outgoing edges."""
        return len(self.adj[node])

    def load_data(self, file_path, node_cap=None):
        """Load Bitcoin OTC dataset with optional node limit."""
        data = pd.read_csv(file_path, names=['from', 'to', 'weight', 'timestamp'], dtype={'from': int, 'to': int, 'weight': int})
        if node_cap:
            data = data[(data['from'] <= node_cap) & (data['to'] <= node_cap)]
        negatives = len(data[data['weight'] < 0])
        print(f"Data: {len(data)} edges, {negatives} negative ({negatives/len(data)*100:.2f}%)")
        for _, row in data.iterrows():
            self.insert_edge(int(row['from']), int(row['to']), int(row['weight']))
        print(f"Graph: {len(self.nodes)} nodes, {len(self.all_edges)} edges")
        return len(self.all_edges)

def bellman_ford(graph, source, trace_file="bellman_ford_trace.txt"):
    """Bellman-Ford algorithm for shortest paths with negative weights."""
    distances = {n: float('inf') for n in graph.nodes}
    distances[source] = 0
    predecessors = {n: None for n in graph.nodes}

    if graph.degree_out(source) == 0:
        print(f"Source {source} has no outgoing edges.")

    with open(trace_file, 'a') as f:
        f.write(f"\nBellman-Ford: Source={source}, Nodes={len(graph.nodes)}\n")
        for i in range(len(graph.nodes) - 1):
            updated = False
            for u, v, w in graph.all_edges:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    predecessors[v] = u
                    updated = True
            f.write(f"Iteration {i + 1}: {'Changes' if updated else 'No changes'}\n")
            if not updated:
                break

        cycle_detected = False
        for u, v, w in graph.all_edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                cycle_detected = True
                f.write(f"Negative cycle on edge ({u}, {v})\n")
                break

    return distances, predecessors, cycle_detected

def trace_path(predecessors, target):
    """Reconstruct shortest path to target."""
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = predecessors[curr]
    return path[::-1]

def save_json(data, filename):
    """Save data to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def save_shortest_paths(distances, predecessors, source, max_nodes, has_cycle):
    """Save and print shortest paths."""
    if has_cycle:
        print(f"\nShortest Paths (Bellman-Ford, max_nodes={max_nodes or 'full'}, source={source})")
        print("Negative cycle detected, paths undefined")
        save_json({"error": "Negative cycle detected"}, f"shortest_paths_bellman_ford_{max_nodes or 'full'}.json")
        return
    paths = [
        {'node': n, 'distance': distances[n], 'path': trace_path(predecessors, n)}
        for n in sorted(distances.keys()) if distances[n] != float('inf')
    ]
    print(f"\nShortest Paths (Bellman-Ford, max_nodes={max_nodes or 'full'}, source={source})")
    print("Node\tDistance\tPath")
    for p in paths:
        path_str = " -> ".join(map(str, p['path']))
        print(f"{p['node']}\t{p['distance']}\t{path_str}")
    for n in sorted(distances.keys()):
        if distances[n] == float('inf'):
            print(f"{n}\tUnreachable\t-")
    save_json(paths, f"shortest_paths_bellman_ford_{max_nodes or 'full'}.json")

def save_execution_time(algorithm, max_nodes, time_taken):
    """Append execution time to file."""
    with open("execution_times.txt", 'a') as f:
        f.write(f"{algorithm} (max_nodes={max_nodes or 'full'}): {time_taken:.4f} seconds\n")

def plot_histogram(distances, algo, nodes, source):
    """Plot histogram of shortest path distances."""
    valid_dists = [d for d in distances.values() if d != float('inf')]
    if not valid_dists:
        print(f"No valid distances for {algo} (nodes={nodes})")
        return
    plt.figure(figsize=(8, 6))
    plt.hist(valid_dists, bins=20, edgecolor='black')
    plt.title(f'{algo} Distance Distribution (Source={source}, Nodes={nodes})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'bellman_ford_dist_{nodes}.png')
    plt.close()

def plot_comparison(dijkstra_data, bellman_data):
    """Plot execution time vs node count as an XY line plot with markers."""
    plt.figure(figsize=(10, 6))
    if dijkstra_data:
        nodes_d = [d['Nodes'] for d in dijkstra_data]
        times_d = [d['Time'] for d in dijkstra_data]
        plt.plot(nodes_d, times_d, 'b-o', label='Dijkstra', linewidth=2, markersize=8)
    if bellman_data:
        nodes_b = [d['Nodes'] for d in bellman_data]
        times_b = [d['Time'] for d in bellman_data]
        plt.plot(nodes_b, times_b, 'r-o', label='Bellman-Ford', linewidth=2, markersize=8)
    plt.title('Execution Time vs Input Size', fontsize=14, pad=10)
    plt.xlabel('Number of Nodes', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('execution_time_analysis.png', dpi=300)
    plt.close()

def run_bellman_ford(graph, source, max_nodes, dataset_path, dijkstra_data):
    """Execute Bellman-Ford and save results."""
    edges = graph.load_data(dataset_path, max_nodes)
    if source not in graph.nodes:
        print(f"Source {source} missing (max_nodes={max_nodes or 'full'})")
        return None
    start = time.perf_counter()
    distances, predecessors, has_cycle = bellman_ford(graph, source)
    end = time.perf_counter()
    duration = end - start

    results = {
        'source': source,
        'max_nodes': max_nodes or 'full',
        'node_count': len(graph.nodes),
        'edge_count': edges,
        'time': duration,
        'negative_cycle': has_cycle,
        'paths': [
            {'node': n, 'distance': distances[n], 'path': trace_path(predecessors, n)}
            for n in sorted(distances.keys()) if distances[n] != float('inf') and not has_cycle
        ],
        'unreachable': [n for n in sorted(distances.keys()) if distances[n] == float('inf')]
    }
    save_json(results, f"bellman_ford_results_{max_nodes or 'full'}.json")
    save_shortest_paths(distances, predecessors, source, max_nodes, has_cycle)
    save_execution_time("Bellman-Ford", max_nodes, duration)
    if not has_cycle:
        plot_histogram(distances, 'Bellman-Ford', len(graph.nodes), source)
    print(f"Bellman-Ford (max_nodes={max_nodes or 'full'}): {duration:.6f}s")
    result = {'Nodes': len(graph.nodes), 'Edges': edges, 'Time': duration}
    plot_comparison(dijkstra_data, [result])
    return result

def main():
    """Run Bellman-Ford on Bitcoin OTC dataset."""
    dataset_path = "soc-sign-bitcoinotc.csv"
    try:
        source = int(input("Enter source node: "))
    except ValueError:
        print("Source must be an integer")
        return

    dijkstra_data = json.load(open("dijkstra_plot_data.json")) if os.path.exists("dijkstra_plot_data.json") else []
    results = []
    for limit in [100, 500, 1000]:
        graph = Graph()
        data = run_bellman_ford(graph, source, limit, dataset_path, dijkstra_data)
        if data:
            results.append(data)

    if input("Run full graph? (y/n): ").lower() == 'y':
        graph = Graph()
        data = run_bellman_ford(graph, source, None, dataset_path, dijkstra_data)
        if data:
            results.append(data)

    if results:
        save_json(results, "bellman_ford_plot_data.json")
        plot_comparison(dijkstra_data, results)
        print("Results saved to JSON")
    else:
        print("No results due to invalid source")

if __name__ == "__main__":
    main()