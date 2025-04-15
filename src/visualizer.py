import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.helper import Track

def draw_graph(
    graph: nx.DiGraph,
    track: Track = None,
    start_state=None,
    highlight_path=None,
    title="Graph Visualization"
):
    fig, ax = plt.subplots(figsize=(10, 10))

    if track:
        # Map cell types to grayscale values
        cell_colors = {
            'O': 0.0,  # Wall (black)
            'G': 0.5,  # Grass (gray)
            ' ': 0.9,  # Road (light)
            'S': 0.7,  # Start
            'F': 0.3,  # Finish
        }

        color_map = np.full((track.rows, track.cols), 1.0)  # default: white

        for r in range(track.rows):
            for c in range(track.cols):
                cell = track.get_cell_type((r, c))
                color_map[r, c] = cell_colors.get(cell, 1.0)

        ax.imshow(color_map, cmap='gray', origin='upper')

    # Position for graph nodes
    pos = {node: (node.col, node.row) for node in graph.nodes}

    # Base graph
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=10,
        node_color='cyan',
        ax=ax,
        alpha=0.6
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=False,
        edge_color='gray',
        width=0.5,
        ax=ax,
        alpha=0.4
    )

    # Start node
    if start_state and start_state in graph:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[start_state],
            node_color='red',
            node_size=100,
            ax=ax,
            label='Start'
        )

    # Path (if any)
    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=highlight_path, node_color='blue', node_size=30, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.axis("equal")
    plt.show()