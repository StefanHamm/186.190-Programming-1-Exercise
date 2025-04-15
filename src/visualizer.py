import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

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
        draw_track_background(ax, track)

    pos = {node: (node.col, node.row) for node in graph.nodes}

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

    if highlight_path:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=highlight_path, node_color='blue', node_size=30, ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

def draw_path_on_track(track, path, title="Path on Track", show_acceleration=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_track_background(ax, track)

    if path:
        if show_acceleration:
            segments = []
            colors = []

            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                segments.append([(start.col, start.row), (end.col, end.row)])

                from math import hypot
                v_start = hypot(start.v_row, start.v_col)
                v_end = hypot(end.v_row, end.v_col)

                if v_end > v_start:
                    colors.append('green')  # accelerating
                elif v_end < v_start:
                    colors.append('red')  # decelerating
                else:
                    colors.append('blue')  # constant speed

            lc = LineCollection(segments, colors=colors, linewidths=2)
            ax.add_collection(lc)
        else:
            xs = [s.col for s in path]
            ys = [s.row for s in path]
            ax.plot(xs, ys, color='blue', linewidth=2, label='Path')

        ax.scatter(path[0].col, path[0].row, color='lime', s=100, label='Start')
        ax.scatter(path[-1].col, path[-1].row, color='red', s=100, label='End')

        # Annotate each step with its index
        for i, state in enumerate(path):
            ax.text(state.col + 0.2, state.row, str(i), fontsize=8, color='black')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    plt.axis("equal")
    plt.show()

def draw_track_background(ax, track):
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

def draw_narrowness_map(track: Track, narrowness_map: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.plasma

    # Walls are shown in black
    masked = np.ma.masked_where(track.track == 'O', narrowness_map)

    ax.imshow(masked, cmap=cmap, origin='upper')
    plt.colorbar(ax.imshow(masked, cmap=cmap), label="Local Width (Free Cells)")
    ax.set_title("Narrowness Heatmap (lower = narrower)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()