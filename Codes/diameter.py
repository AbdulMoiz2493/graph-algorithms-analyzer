import networkx as nx
import csv
from itertools import islice
import os
import time

def load_bitcoin_otc_data(file_path):
    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found in the current directory ({os.getcwd()}). "
                               "Please ensure the file 'soc-sign-bitcoin-otc.csv' is in the same directory as this script.")
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Read CSV data
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        
        has_weight = False
        for row in csv_reader:
            source, target = row[0], row[1]
            # Check if dataset has weights (rating column)
            weight = float(row[2]) if len(row) > 2 else None
            if weight is not None:
                has_weight = True
                G.add_edge(source, target, weight=weight)
            else:
                G.add_edge(source, target)
    
    return G, has_weight

def make_weights_positive(G):
    # Create a new graph with positive weights
    G_positive = G.copy()
    for u, v, d in G_positive.edges(data=True):
        if 'weight' in d:
            d['weight'] = abs(d['weight'])  # Use absolute value
    return G_positive

def calculate_diameter(G, is_weighted, is_directed):
    def eccentricity(G, v, weight=None):
        # Compute shortest path lengths from node v to all reachable nodes
        if is_weighted and weight:
            try:
                # Use Bellman-Ford for negative weights
                lengths = nx.single_source_bellman_ford_path_length(G, v, weight=weight)
                return max(lengths.values()) if lengths else float('inf')
            except nx.NetworkXUnbounded:
                return float('inf')  # Negative cycle detected
        else:
            # Use BFS for unweighted graphs
            lengths = nx.single_source_shortest_path_length(G, v)
            return max(lengths.values()) if lengths else float('inf')

    # Handle different graph types
    result = []
    if is_directed:
        if is_weighted:
            # Check for negative cycles
            has_negative_cycle = nx.negative_edge_cycle(G, weight='weight')
            if has_negative_cycle:
                result.append("Negative cycle detected in the graph.")
                result.append("Converting weights to positive using absolute values.")
                # Convert weights to positive and recompute diameter
                G_positive = make_weights_positive(G)
                try:
                    # Compute eccentricity for each node in the positive-weighted graph
                    ecc = {n: eccentricity(G_positive, n, weight='weight') for n in islice(G_positive.nodes(), 1000)}
                    diameter = max(ecc.values())
                    return diameter if diameter != float('inf') else None, "Directed, Weighted (Converted to Positive Weights)", result
                except nx.NetworkXError:
                    result.append("Graph is not strongly connected.")
                    return None, "Directed, Weighted (Converted to Positive Weights)", result
            else:
                # No negative cycles, compute diameter normally
                try:
                    ecc = {n: eccentricity(G, n, weight='weight') for n in islice(G.nodes(), 1000)}
                    diameter = max(ecc.values())
                    return diameter if diameter != float('inf') else None, "Directed, Weighted", result
                except nx.NetworkXError:
                    result.append("Graph is not strongly connected.")
                    return None, "Directed, Weighted", result
        else:
            # For unweighted directed graph
            try:
                diameter = nx.diameter(G)
                return diameter, "Directed, Unweighted", result
            except nx.NetworkXError:
                result.append("Graph is not strongly connected.")
                return None, "Directed, Unweighted", result
    else:
        # Convert to undirected graph
        G_undirected = G.to_undirected()
        if is_weighted:
            # Check for negative cycles
            has_negative_cycle = nx.negative_edge_cycle(G, weight='weight')
            if has_negative_cycle:
                result.append("Negative cycle detected in the graph.")
                result.append("Converting weights to positive using absolute values.")
                # Convert weights to positive and recompute diameter
                G_positive = make_weights_positive(G_undirected)
                try:
                    ecc = {n: eccentricity(G_positive, n, weight='weight') for n in islice(G_positive.nodes(), 1000)}
                    diameter = max(ecc.values())
                    return diameter if diameter != float('inf') else None, "Undirected, Weighted (Converted to Positive Weights)", result
                except nx.NetworkXError:
                    result.append("Graph is not connected.")
                    return None, "Undirected, Weighted (Converted to Positive Weights)", result
            else:
                # No negative cycles, compute diameter normally
                try:
                    ecc = {n: eccentricity(G_undirected, n, weight='weight') for n in islice(G_undirected.nodes(), 1000)}
                    diameter = max(ecc.values())
                    return diameter if diameter != float('inf') else None, "Undirected, Weighted", result
                except nx.NetworkXError:
                    result.append("Graph is not connected.")
                    return None, "Undirected, Weighted", result
        else:
            # For unweighted undirected graph
            try:
                diameter = nx.diameter(G_undirected)
                return diameter, "Undirected, Unweighted", result
            except nx.NetworkXError:
                result.append("Graph is not connected.")
                return None, "Undirected, Unweighted", result

def main():
    # Path to the local Bitcoin OTC dataset CSV file
    # Assumes soc-sign-bitcoin-otc.csv is in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "soc-sign-bitcoinotc.csv")
    
    # Load the dataset
    G, has_weight = load_bitcoin_otc_data(file_path)
    
    # Determine graph properties
    is_directed = nx.is_directed(G)
    is_weighted = has_weight
    
    # Measure execution time
    start_time = time.time()
    diameter, graph_type, result_messages = calculate_diameter(G, is_weighted, is_directed)
    execution_time = time.time() - start_time
    
    # Print results
    print(f"Graph Type: {graph_type}")
    for msg in result_messages:
        print(msg)
    if diameter is not None:
        print(f"Diameter: {diameter}")
    else:
        print("Diameter: Cannot be computed due to connectivity issues")
    print(f"Execution Time: {execution_time:.4f} seconds")
    
    # Store execution time in execution_time.txt
    with open(os.path.join(script_dir, "execution_time.txt"), 'w') as f:
        f.write(f"Execution Time: {execution_time:.4f} seconds\n")
    
    # Store results in graph_results.txt
    with open(os.path.join(script_dir, "graph_results.txt"), 'w') as f:
        f.write(f"Graph Type: {graph_type}\n")
        for msg in result_messages:
            f.write(f"{msg}\n")
        if diameter is not None:
            f.write(f"Diameter: {diameter}\n")
        else:
            f.write("Diameter: Cannot be computed due to connectivity issues\n")

if __name__ == "__main__":
    main()