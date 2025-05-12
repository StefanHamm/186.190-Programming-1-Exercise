from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button # Import Button widget
import pygame
import time
from math import hypot

# Assuming src.helper.Track is defined elsewhere, e.g.:
# from src.helper import Track
# For placeholder purposes if Track is not available:
class Track:
    def __init__(self, rows, cols, track_data=None):
        self.rows = rows
        self.cols = cols
        if track_data is None:
            # Example track data if none is provided
            self.track = [['S', ' ', 'F']]
            self.rows = len(self.track)
            self.cols = len(self.track[0])
        else:
            self.track = track_data

    def get_cell_type(self, pos):
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.track[r][c]
        return 'O' # Wall if out of bounds

class PlotManager:
    def __init__(self):
        self.plot_recipes = []    # Stores dictionaries: {'method': render_method, 'args': args, 'kwargs': kwargs, 'title': title}
        self.current_plot_index = 0
        self.fig = None           # The single figure used for display
        self.nav_buttons = []     # To keep references to button objects, preventing garbage collection

    @staticmethod
    def _draw_track_background(ax, track: Track):
        cell_colors = {
            'O': 0.0,  # Wall (black)
            'G': 0.5,  # Grass (gray)
            ' ': 0.9,  # Road (light)
            'S': 0.7,  # Start
            'F': 0.3,  # Finish
        }
        color_map = np.full((track.rows, track.cols), 1.0)  # default: white
        for r_idx in range(track.rows):
            for c_idx in range(track.cols):
                cell = track.get_cell_type((r_idx, c_idx))
                color_map[r_idx, c_idx] = cell_colors.get(cell, 1.0)
        # Ensure consistent color mapping for imshow
        ax.imshow(color_map, cmap='gray', origin='upper', vmin=0.0, vmax=1.0)

    def _add_plot_recipe(self, render_method, display_title, *args, **kwargs):
        """
        Adds a recipe for creating a plot.
        render_method: The internal method to call for rendering (e.g., self._render_graph_plot).
        display_title: The title to be shown for this plot (e.g., in window title or button).
        *args, **kwargs: Arguments to be passed to the render_method.
        """
        self.plot_recipes.append({
            'method': render_method,
            'args': args,
            'kwargs': kwargs,
            'title': display_title
        })

    # --- Internal Rendering Methods ---
    # Each takes `fig` as its first argument (after self) and draws on it.
    
    def _render_graph_plot(self, fig, graph: nx.DiGraph, track: Track = None, start_state=None, highlight_path=None, title="Graph Visualization"):
        ax = fig.add_subplot(111) # Add a default axes to the figure

        if track:
            PlotManager._draw_track_background(ax, track)

        pos = {node: (node.col, node.row) for node in graph.nodes}

        nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='cyan', ax=ax, alpha=0.6)
        nx.draw_networkx_edges(graph, pos, arrows=False, edge_color='gray', width=0.5, ax=ax, alpha=0.4)

        edge_attributes = nx.get_edge_attributes(graph, 'weight')
        edge_labels = {}
        for edge, weight in edge_attributes.items():
            if isinstance(weight, float):
                edge_labels[edge] = f"{weight:.1f}"
            else:
                edge_labels[edge] = str(weight)

        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, font_color='black', ax=ax)

        if start_state and start_state in graph:
            nx.draw_networkx_nodes(graph, pos, nodelist=[start_state], node_color='red', node_size=100, ax=ax, label='Start')

        if highlight_path:
            path_edges = list(zip(highlight_path, highlight_path[1:]))
            nx.draw_networkx_nodes(graph, pos, nodelist=highlight_path, node_color='blue', node_size=30, ax=ax)
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='blue', width=2, ax=ax)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles)) # Remove duplicate labels
            ax.legend(by_label.values(), by_label.keys())
            
        ax.set_aspect('equal', adjustable='box')

    def create_graph_plot(self, graph: nx.DiGraph, track: Track = None, start_state=None, highlight_path=None, title="Graph Visualization"):
        # The 'title' argument here is used both as the plot's title and the display title for navigation
        self._add_plot_recipe(self._render_graph_plot, title, graph, track, start_state, highlight_path, title=title)

    def _render_path_on_track_plot(self, fig, track: Track, path, title="Path on Track", show_acceleration=False):
        ax = fig.add_subplot(111)
        PlotManager._draw_track_background(ax, track)

        if path:
            if show_acceleration:
                segments = []
                colors = []
                for i in range(len(path) - 1):
                    start_node = path[i]
                    end_node = path[i + 1]
                    segments.append([(start_node.col, start_node.row), (end_node.col, end_node.row)])

                    v_start = hypot(start_node.v_row, start_node.v_col)
                    v_end = hypot(end_node.v_row, end_node.v_col)

                    if v_end > v_start: colors.append('green')
                    elif v_end < v_start: colors.append('red')
                    else: colors.append('blue')
                
                lc = LineCollection(segments, colors=colors, linewidths=2)
                ax.add_collection(lc)
            else:
                xs = [s.col for s in path]
                ys = [s.row for s in path]
                ax.plot(xs, ys, color='blue', linewidth=2, label='Path')

            ax.scatter(path[0].col, path[0].row, color='lime', s=100, label='Start')
            ax.scatter(path[-1].col, path[-1].row, color='red', s=100, label='End')

            for i, state in enumerate(path):
                ax.text(state.col + 0.2, state.row, str(i), fontsize=8, color='black')

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        if path: ax.legend()
        ax.set_aspect('equal', adjustable='box')

    def create_path_on_track_plot(self, track: Track, path, title="Path on Track", show_acceleration=False):
        self._add_plot_recipe(self._render_path_on_track_plot, title, track, path, title=title, show_acceleration=show_acceleration)

    def _render_map_data_plot(self, fig, track: Track, map_data: np.ndarray, title: str, cmap_label: str):
        ax = fig.add_subplot(111)
        cmap = plt.cm.plasma
        wall_mask = np.array([[cell == 'O' for cell in row] for row in track.track])
        masked_data = np.ma.masked_where(wall_mask, map_data)
        im = ax.imshow(masked_data, cmap=cmap, origin='upper')
        fig.colorbar(im, ax=ax, label=cmap_label, shrink=0.8) # shrink to make space
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    def create_map_data_plot(self, track: Track, map_data: np.ndarray, title: str, cmap_label: str):
        self._add_plot_recipe(self._render_map_data_plot, title, track, map_data, title=title, cmap_label=cmap_label)

    # --- Display Logic for Single Figure Interactive Viewer ---
    def _display_selected_plot(self):
        if not self.plot_recipes or self.fig is None:
            return

        self.fig.clf() # Clear the entire figure for the new plot

        recipe = self.plot_recipes[self.current_plot_index]
        render_method = recipe['method'] # This is a bound method of self
        args = recipe['args']
        kwargs = recipe['kwargs']
        
        # Call the render method (e.g., self._render_graph_plot(self.fig, *args, **kwargs))
        render_method(self.fig, *args, **kwargs)

        current_title = recipe['title']
        if self.fig.canvas.manager is not None: # Check if a window manager exists (e.g. not for Agg backend)
             self.fig.canvas.manager.set_window_title(f"Plot Viewer: {current_title} ({self.current_plot_index + 1}/{len(self.plot_recipes)})")
        
        self._add_navigation_buttons()
        
        # Adjust layout: rect=[left, bottom, right, top] in normalized figure coordinates
        # Reserve 6% at bottom for buttons, 5% at top for main title area.
        self.fig.tight_layout(rect=[0, 0.06, 1, 0.95]) 
        
        self.fig.canvas.draw_idle() # Redraw the figure

    def _add_navigation_buttons(self):
        # Clear any old button objects (though fig.clf() should handle their axes)
        self.nav_buttons.clear()

        button_height = 0.04
        button_width = 0.09 
        horizontal_pos_start = 0.99 - button_width # Start from right edge
        vertical_pos = 0.01 # Bottom margin
        
        # Next Button
        ax_next = self.fig.add_axes([horizontal_pos_start, vertical_pos, button_width, button_height])
        btn_next = Button(ax_next, 'Next')
        btn_next.on_clicked(self._on_next)
        self.nav_buttons.append(btn_next) # Keep reference

        # Previous Button (to the left of Next)
        ax_prev = self.fig.add_axes([horizontal_pos_start - button_width - 0.01, vertical_pos, button_width, button_height])
        btn_prev = Button(ax_prev, 'Previous')
        btn_prev.on_clicked(self._on_prev)
        self.nav_buttons.append(btn_prev)
        
        # Disable buttons if at the start/end of the plot list
        if self.current_plot_index == 0:
            btn_prev.set_active(False)
        if self.current_plot_index == len(self.plot_recipes) - 1:
            btn_next.set_active(False)

    def _on_next(self, event):
        if self.current_plot_index < len(self.plot_recipes) - 1:
            self.current_plot_index += 1
            self._display_selected_plot()

    def _on_prev(self, event):
        if self.current_plot_index > 0:
            self.current_plot_index -= 1
            if self.current_plot_index == 0:
                self.current_plot_index = len(self.plot_recipes) - 1
            self._display_selected_plot()

    def show_plots(self):
        if not self.plot_recipes:
            print("No plots to show.")
            return

        # Create the figure if it doesn't exist or if the existing one was closed
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(10, 10)) # You can make figsize configurable
        
        # Ensure current_plot_index is valid (e.g., if plots were cleared then new ones added)
        if self.current_plot_index >= len(self.plot_recipes) or self.current_plot_index < 0:
            self.current_plot_index = 0
            
        self._display_selected_plot() # Display the initial or current plot
        plt.show() # Show the single figure and start Matplotlib event loop

    def clear_plots(self):
        """Clears all stored plot recipes and closes the figure."""
        self.plot_recipes = []
        self.current_plot_index = 0
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        self.nav_buttons.clear() # Clear button references
        # print("Plot recipes cleared and figure closed.") # Optional: print confirmation


# The Pygame Visualizer class remains largely the same
class Visualizer:
    def __init__(self, track, width=800, height=800):
        pygame.init()
        self.track = track
        self.screen_size = (width, height)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Racetrack Visualization")
        self.clock = pygame.time.Clock()
        self.cell_width = width // track.cols
        self.cell_height = height // track.rows

    def draw(self, visited, open_heap, path):
        self.screen.fill((255, 255, 255))  # White background

        for r in range(self.track.rows):
            for c in range(self.track.cols):
                rect = pygame.Rect(
                    c * self.cell_width,
                    r * self.cell_height,
                    self.cell_width,
                    self.cell_height
                )
                cell_type = self.track.get_cell_type((r, c))
                if cell_type == 'O': color = (0, 0, 0)
                elif cell_type == 'G': color = (150, 150, 150)
                elif cell_type == 'S': color = (0, 255, 255)
                elif cell_type == 'F': color = (255, 215, 0)
                else: color = (220, 220, 220)
                pygame.draw.rect(self.screen, color, rect)

        for state in visited: # Assuming state has .col and .row
            rect = pygame.Rect(state.col * self.cell_width, state.row * self.cell_height, self.cell_width, self.cell_height)
            pygame.draw.rect(self.screen, (100, 100, 255), rect, 1) # Border for visited

        for item in open_heap:
            if len(item) >= 4 and hasattr(item[3], 'col') and hasattr(item[3], 'row'): 
                state = item[3]
                rect = pygame.Rect(state.col * self.cell_width, state.row * self.cell_height, self.cell_width, self.cell_height)
                pygame.draw.rect(self.screen, (0, 255, 0), rect, 1) 

        if path:
            for state in path:
                rect_path = pygame.Rect(
                    state.col * self.cell_width + self.cell_width // 4,
                    state.row * self.cell_height + self.cell_height // 4,
                    self.cell_width // 2,
                    self.cell_height // 2
                )
                pygame.draw.rect(self.screen, (255, 0, 0), rect_path) 

        pygame.display.flip()
        self.clock.tick(60)

    def quit(self):
        time.sleep(2) 
        pygame.quit()
        
def animate_paths_pygame(track: Track, paths_list: list,
                         screen_width: int = 800, screen_height: int = 800,
                         delay_ms: int = 500, path_color=(255, 0, 0), path_thickness=3,
                         loop_animation: bool = False):
    """
    Animates a list of paths on a track using Pygame in a new window.
    Each path is displayed for `delay_ms` milliseconds.

    Args:
        track (Track): The track object with .rows, .cols, and .get_cell_type().
        paths_list (list): A list of paths. Each path is a list of state objects.
                           Each state object must have .row and .col attributes.
        screen_width (int): Width of the Pygame window.
        screen_height (int): Height of the Pygame window.
        delay_ms (int): Time in milliseconds to display each path. Set to 0 for manual stepping.
        path_color (tuple): RGB color for the path.
        path_thickness (int): Thickness of the path line.
        loop_animation (bool): If True, the animation loops after the last path (only if delay_ms > 0).
    """
    if not paths_list:
        print("No paths to animate.")
        return

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Path Animation (ESC to quit, Left/Right to step)")
    clock = pygame.time.Clock()

    cell_width = screen_width // track.cols
    cell_height = screen_height // track.rows

    try:
        font = pygame.font.SysFont(None, 30) # Font for path index
    except pygame.error: # Fallback if SysFont fails (e.g. minimal Pygame install)
        font = pygame.font.Font(None, 30)


    running = True
    current_path_idx = 0
    last_time = pygame.time.get_ticks()

    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_RIGHT:
                    current_path_idx = (current_path_idx + 1) % len(paths_list)
                    last_time = now # Reset timer for manual step
                if event.key == pygame.K_LEFT:
                    current_path_idx = (current_path_idx - 1 + len(paths_list)) % len(paths_list)
                    last_time = now # Reset timer for manual step

        # --- Automatic path advancement ---
        if delay_ms > 0 and (now - last_time > delay_ms):
            if loop_animation or current_path_idx < len(paths_list) - 1:
                current_path_idx = (current_path_idx + 1) % len(paths_list)
            last_time = now

        # --- Drawing ---
        screen.fill((255, 255, 255))  # White background

        # Draw track
        for r in range(track.rows):
            for c in range(track.cols):
                rect = pygame.Rect(c * cell_width, r * cell_height, cell_width, cell_height)
                cell_type = track.get_cell_type((r, c))
                color_map = {
                    'O': (0, 0, 0), 'G': (150, 150, 150), ' ': (220, 220, 220),
                    'S': (0, 255, 255), 'F': (255, 215, 0)
                }
                color = color_map.get(cell_type, (100, 100, 100))
                pygame.draw.rect(screen, color, rect)
                # pygame.draw.rect(screen, (50,50,50), rect, 1) # Optional grid lines

        # Draw current path
        if paths_list and 0 <= current_path_idx < len(paths_list):
            path = paths_list[current_path_idx]
            if path and len(path) >= 2:
                points = []
                for state in path: # Assuming state has .row and .col
                    x = state.col * cell_width + cell_width // 2
                    y = state.row * cell_height + cell_height // 2
                    points.append((x, y))
                pygame.draw.lines(screen, path_color, False, points, path_thickness)
            elif path and len(path) == 1:
                state = path[0]
                x = state.col * cell_width + cell_width // 2
                y = state.row * cell_height + cell_height // 2
                pygame.draw.circle(screen, path_color, (x,y), path_thickness * 2)

            path_text_surface = font.render(f"Path: {current_path_idx + 1}/{len(paths_list)}", True, (10, 10, 10))
            screen.blit(path_text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(30) # Limit FPS

    pygame.quit()

def draw_graph_on_track(graph, track, title="Graph on Track"):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Background: grayscale racetrack
    color_map = np.full((track.rows, track.cols), 1.0)
    for r in range(track.rows):
        for c in range(track.cols):
            cell = track.get_cell_type((r, c))
            if cell == 'O':
                color_map[r, c] = 0.0  # Wall
            elif cell == 'G':
                color_map[r, c] = 0.5  # Grass
            elif cell == 'F':
                color_map[r, c] = 0.3  # Finish
            elif cell == 'S':
                color_map[r, c] = 0.7  # Start
            else:
                color_map[r, c] = 0.9  # Road

    ax.imshow(color_map, cmap='gray', origin='upper')

    # Graph overlay
    pos = {node: (node.col, node.row) for node in graph.nodes()}
    nx.draw_networkx_nodes(graph, pos, node_size=10, node_color='cyan', ax=ax, alpha=0.6)
    nx.draw_networkx_edges(graph, pos, edge_color='orange', alpha=0.3, arrows=False, ax=ax)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

# Example Usage (updated to use PlotManager)
if __name__ == '__main__':
    # Node class for graph plotting (must be hashable for NetworkX)
    class Node:
        def __init__(self, r, c, name=""):
            self.row = r
            self.col = c
            self.name = name
        def __repr__(self): return self.name if self.name else f"N({self.row},{self.col})"
        def __lt__(self, other): return (self.row, self.col) < (other.row, other.col) # For sorting
        def __hash__(self): return hash((self.row, self.col, self.name)) # Hashable
        def __eq__(self, other): # Equality check
            return isinstance(other, self.__class__) and \
                   (self.row, self.col, self.name) == (other.row, other.col, other.name)

    # State class for path plotting (if show_acceleration=True)
    class PathState(Node):
        def __init__(self, r, c, name="", v_r=0.0, v_c=0.0):
            super().__init__(r, c, name)
            self.v_row = v_r
            self.v_col = v_c

    # --- Create a PlotManager instance ---
    plot_manager = PlotManager()

    # --- Plot 1: Simple Graph ---
    G_simple = nx.DiGraph()
    n1 = Node(0, 0, "N1")
    n2 = Node(0, 1, "N2")
    n3 = Node(1, 1, "N3")
    G_simple.add_nodes_from([n1, n2, n3])
    G_simple.add_edge(n1, n2, weight=1.2)
    G_simple.add_edge(n2, n3, weight=3.4)
    simple_track_data = [['S', ' '], [' ', 'F']]
    simple_track = Track(rows=2, cols=2, track_data=simple_track_data)
    
    print("Creating Plot 1: Simple Graph")
    plot_manager.create_graph_plot(G_simple, track=simple_track, start_state=n1, title="Simple Graph Demo")

    # --- Plot 2: Grid Graph with Path ---
    grid_size = 4
    G_grid = nx.DiGraph()
    nodes_map = {}
    for r in range(grid_size):
        for c in range(grid_size):
            node = Node(r, c, name=f"G({r},{c})") 
            nodes_map[(r,c)] = node
            G_grid.add_node(node)

    np.random.seed(42) 
    for r_idx in range(grid_size):
        for c_idx in range(grid_size):
            curr_n = nodes_map[(r_idx, c_idx)]
            if c_idx + 1 < grid_size: 
                G_grid.add_edge(curr_n, nodes_map[(r_idx, c_idx + 1)], weight=round(np.random.rand()+0.1, 1))
            if r_idx + 1 < grid_size: 
                G_grid.add_edge(curr_n, nodes_map[(r_idx + 1, c_idx)], weight=round(np.random.rand()+0.1, 1))
    
    start_node_grid = nodes_map.get((0,0))
    grid_path = [nodes_map[(0,0)], nodes_map[(0,1)], nodes_map[(1,1)], nodes_map[(1,2)]] if grid_size >=3 else None
    
    grid_track_data = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    if grid_size > 0: grid_track_data[0][0] = 'S'
    if grid_size > 1: grid_track_data[grid_size-1][grid_size-1] = 'F'
    detailed_track = Track(rows=grid_size, cols=grid_size, track_data=grid_track_data)

    print("Creating Plot 2: Grid Graph with Path")
    plot_manager.create_graph_plot(G_grid, track=detailed_track, start_state=start_node_grid, 
                                   highlight_path=grid_path, title=f"{grid_size}x{grid_size} Grid Graph")

    # --- Plot 3: Path on Track with Acceleration ---
    path_for_accel = [
        PathState(0, 0, v_r=0, v_c=0), PathState(0, 1, v_r=0, v_c=1), 
        PathState(1, 1, v_r=1, v_c=1), PathState(1, 2, v_r=1, v_c=0), 
        PathState(2, 2, v_r=0, v_c=0) 
    ]
    print("Creating Plot 3: Path on Track with Acceleration")
    plot_manager.create_path_on_track_plot(detailed_track, path_for_accel, 
                                           title="Path with Acceleration", show_acceleration=True)

    # --- Plot 4: Map Data Display (Heatmap) ---
    example_map_data = np.random.rand(grid_size, grid_size) * 100
    if grid_size >=2:
      example_map_data[0,0] = 0 
      example_map_data[1,1] = 100
    
    print("Creating Plot 4: Map Data Display")
    plot_manager.create_map_data_plot(detailed_track, example_map_data, 
                                      title="Example Heatmap", cmap_label="Value")

    # --- Show all plots in the interactive viewer ---
    print("Showing all created plots...")
    plot_manager.show_plots()
    
    print("Example finished. Close the Matplotlib window to exit.")