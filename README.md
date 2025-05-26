# 📊 Graph Algorithms Analyzer

Efficient implementation and analysis of core graph algorithms including:
- Single Source Shortest Path (Dijkstra, Bellman-Ford)
- Minimum Spanning Trees (Prim's, Kruskal's)
- Graph Traversals (BFS, DFS)
- Diameter of Graph
- Cycle Detection
- Average Degree Calculation

Designed to handle large-scale real-world graphs (1000+ nodes), the project includes full stack/queue traces, execution times, and comparison plots. Tested on SNAP datasets from Stanford.

---

## 🔍 Features

- Efficient algorithms tailored for large graphs
- Stack and queue traces stored in separate files
- Execution time logging for all implementations
- Clean output for shortest paths, MST, diameter, etc.
- Tested using SNAP datasets (1000+ nodes)
- Optional graph visualizations
- CLI-based node selection for BFS/DFS and shortest path

---

## 📁 File Structure

```

graph-algorithms-analyzer/
│
├── bellmanFord.py                 # Bellman-Ford algorithm
├── bfs.py                         # Breadth-First Search
├── cycle.py                       # Cycle detection
├── dataset.csv                    # Optional sample dataset
├── degree.py                      # Average degree calculation
├── dfs.py                         # Depth-First Search
├── diameter.py                    # Graph diameter
├── dijkstra.py                    # Dijkstra’s shortest path
├── kruskal.py                     # Kruskal’s MST
├── prims.py                       # Prim’s MST
└── soc-sign-bitcoinotc.csv        # Main SNAP dataset

```

---

## 🧪 Algorithms Implemented

| Problem                          | Algorithms Used                    |
|----------------------------------|------------------------------------|
| Single Source Shortest Path      | Dijkstra, Bellman-Ford             |
| Minimum Spanning Tree            | Prim’s, Kruskal’s                  |
| Traversals                       | BFS, DFS                           |
| Diameter of Graph                | All-Pairs Shortest Path (Max)      |
| Cycle Detection                  | DFS-based detection                |
| Average Degree                   | Total Degree / Number of Nodes     |

---

## 🧠 Time Complexities

| Algorithm        | Best Case | Average Case | Worst Case |
|------------------|-----------|--------------|-------------|
| Dijkstra (Heap)  | O((V + E) log V) | Same      | Same       |
| Bellman-Ford     | O(VE)     | Same         | Same        |
| Prim’s (Heap)    | O((V + E) log V) | Same     | Same        |
| Kruskal’s        | O(E log V) | Same         | Same        |
| BFS / DFS        | O(V + E)   | Same         | Same        |
| Diameter         | O(V*(V + E)) | Same      | Same        |
| Cycle Detection  | O(V + E)   | Same         | Same        |

---

## 📈 Performance Analysis

- Execution time measured using `time` module
- Graphs plotted to show time vs. graph size
- Dense vs. sparse comparison
- Directed vs. undirected effect
- Stack vs. queue structure impact

Example plotting library:
```bash
pip install matplotlib plotly
```

---

## 📁 Output Structure (Expected)

Example output files stored during execution:

```
/output
  ├── dijkstra_trace.txt
  ├── bellman_ford_trace.txt
  ├── bfs_trace.txt
  ├── dfs_trace.txt
  ├── mst_kruskal_result.txt
  ├── mst_prim_result.txt
  ├── diameter_trace.txt
  ├── cycle_detection_result.txt
  ├── average_degree.txt
  └── execution_times.csv
```

---

## ⚙️ Installation & Running

### Requirements

```bash
pip install networkx matplotlib pandas
```

### Run Example

```bash
python dijkstra.py
```

Ensure `soc-sign-bitcoinotc.csv` or `dataset.csv` is correctly referenced inside the scripts.

---

## 🧾 Dataset Info

Main dataset used:
**soc-sign-bitcoinotc.csv** – from Stanford SNAP collection.
Contains over 3,000 nodes and 35,000+ edges (weighted, directed social graph).

Dataset link:
🔗 [https://snap.stanford.edu/data/soc-sign-bitcoinotc.html](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html)

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

---

## 📬 Contact

- **Author:** Abdul Moiz
- **Email:** [abdulmoiz8895@gmail.com](mailto:abdulmoiz8895@gmail.com)
- **GitHub:** [github.com/AbdulMoiz2493](https://github.com/AbdulMoiz2493)

