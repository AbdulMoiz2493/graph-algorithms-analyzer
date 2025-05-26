import heapq
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import defaultdict

class Graph:
    """Directed, weighted graph for shortest path computation."""
    def __init__(self):
        self.edges = defaultdict(list)  # {node: [(neighbor, weight), ...]}
        self.node_set = set()

    def add_edge(self, src, dst, wt):
        """Add a directed edge with weight."""
        self.edges[src].append((dst, wt))
        self.node_set.add(src)
        self.node_set.add(dst)

    def get_adjacent(self, node):
        """Return (neighbor, weight) pairs for a node."""
        return self.edges.get(node, [])

    def out_degree(self, node):
        """Count outgoing edges."""
        return len(self.edges[node])

    def load_data(self, file_path, max_nodes=None):
        """Load Bitcoin OTC dataset, optionally limiting nodes."""
        df = pd.read_csv(file_path, names=['source', 'target', 'rating', 'time'], dtype={'source': int, 'target': int, 'rating': int})
        if max_nodes:
            df = df[(df['source'] <= max_nodes) & (df['target'] <= max_nodes)]
        neg_count = len(df[df['rating'] < 0])
        print(f"Loaded {len(df)} edges, {neg_count} negative ({neg_count/len(df)*100:.2f}%)")
        edge_count = 0
        for _, row in df.iterrows():
            src, tgt, wt = int(row['source']), int(row['target']), int(row['rating'])
            self.add_edge(src, tgt, wt + 10)  # Shift weights
            edge_count += 1
        print(f"Graph: {len(self.node_set)} nodes, {edge_count} edges")
        return edge_count

def dijkstra(graph, source, log_file="dijkstra_log.txt"):
    """Dijkstra's algorithm for shortest paths with non-negative weights."""
    distances = {n: float('inf') for n in graph.node_set}
    distances[source] = 0
    predecessors = {n: None for n in graph.node_set}
    path_counts = {n: 0 for n in graph.node_set}
    pq = [(0, source)]
    visited = set()

    if graph.out_degree(source) == 0:
        print(f"Source {source} has no outgoing edges.")

    with open(log_file, 'a') as f:
        f.write(f"\nDijkstra: Source={source}, Nodes={len(graph.node_set)}\n")
        while pq:
            dist, node = heapq.heappop(pq)
            f.write(f"Popped {node} (dist={dist})\n")
            if node in visited:
                continue
            visited.add(node)
            for neighbor, weight in graph.get_adjacent(node):
                if neighbor in visited:
                    continue
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = node
                    path_counts[neighbor] = path_counts[node] + 1
                    heapq.heappush(pq, (new_dist, neighbor))
                    f.write(f"Updated {neighbor}: dist={new_dist}\n")

    adjusted = {n: distances[n] - 10 * path_counts[n] if distances[n] != float('inf') else float('inf') for n in distances}
    return adjusted, predecessors, path_counts

def build_path(predecessors, target):
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

def save_shortest_paths(distances, predecessors, source, max_nodes):
    """Save and print shortest paths."""
    paths = [
        {'node': n, 'distance': distances[n], 'path': build_path(predecessors, n)}
        for n in sorted(distances.keys()) if distances[n] != float('inf')
    ]
    print(f"\nShortest Paths (Dijkstra, max_nodes={max_nodes or 'full'}, source={source})")
    print("Node\tDistance\tPath")
    for p in paths:
        path_str = " -> ".join(map(str, p['path']))
        print(f"{p['node']}\t{p['distance']}\t{path_str}")
    for n in sorted(distances.keys()):
        if distances[n] == float('inf'):
            print(f"{n}\tUnreachable\t-")
    save_json(paths, f"shortest_paths_dijkstra_{max_nodes or 'full'}.json")

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
    plt.title(f'{algo} Shortest Paths (Source={source}, Nodes={nodes})')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'dijkstra_dist_{nodes}.png')
    plt.close()

def run_dijkstra(graph, source, max_nodes, dataset_path):
    """Execute Dijkstra and save results."""
    edges = graph.load_data(dataset_path, max_nodes)
    if source not in graph.node_set:
        print(f"Source {source} missing (max_nodes={max_nodes or 'full'})")
        return None
    start = time.perf_counter()
    distances, predecessors, path_counts = dijkstra(graph, source)
    end = time.perf_counter()
    duration = end - start

    results = {
        'source': source,
        'max_nodes': max_nodes or 'full',
        'node_count': len(graph.node_set),
        'edge_count': edges,
        'time': duration,
        'paths': [
            {'node': n, 'distance': distances[n], 'path': build_path(predecessors, n)}
            for n in sorted(distances.keys()) if distances[n] != float('inf')
        ],
        'unreachable': [n for n in sorted(distances.keys()) if distances[n] == float('inf')]
    }
    save_json(results, f"dijkstra_results_{max_nodes or 'full'}.json")
    save_shortest_paths(distances, predecessors, source, max_nodes)
    save_execution_time("Dijkstra", max_nodes, duration)
    plot_histogram(distances, 'Dijkstra', len(graph.node_set), source)
    print(f"Dijkstra (max_nodes={max_nodes or 'full'}): {duration:.6f}s")
    return {'Nodes': len(graph.node_set), 'Edges': edges, 'Time': duration}

def main():
    """Run Dijkstra on Bitcoin OTC dataset."""
    dataset_path = "soc-sign-bitcoinotc.csv"
    try:
        source = int(input("Enter source node: "))
    except ValueError:
        print("Source must be an integer")
        return

    results = []
    for limit in [100, 500, 1000]:
        graph = Graph()
        data = run_dijkstra(graph, source, limit, dataset_path)
        if data:
            results.append(data)

    if input("Run full graph? (y/n): ").lower() == 'y':
        graph = Graph()
        data = run_dijkstra(graph, source, None, dataset_path)
        if data:
            results.append(data)

    if results:
        save_json(results, "dijkstra_plot_data.json")
        print("Results saved to JSON")
    else:
        print("No results due to invalid source")

if __name__ == "__main__":
    main()