# in this file we run the construction heuristcs of the tracks
from collections import deque

from matplotlib import pyplot as plt

from helper import loadTrack, displayTrack, run_visualization_in_docker, Track
from src.helper import bresenham_line
from src.visualizer import draw_narrowness_map
from visualizer import draw_graph, draw_path_on_track
import networkx as nx
import numpy as np
import argparse

from src.state import CarState

def compute_narrowness_map(track: Track, radius: int = 1) -> np.ndarray:
    narrowness_map = np.full((track.rows, track.cols), np.nan)

    for r in range(track.rows):
        for c in range(track.cols):
            if track.get_cell_type((r, c)) == 'O':
                continue  # skip walls entirely

            free = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if track.is_valid_coordinate((nr, nc)):
                        cell = track.get_cell_type((nr, nc))
                        if cell != 'O':  # count only non-wall neighbors
                            free += 1

            narrowness_map[r, c] = free

    return narrowness_map

def is_valid_transition(track: Track, from_state: CarState, to_state: CarState) -> bool:
    if not track.is_valid_coordinate(to_state.position()):
        return False

    if from_state.position() == to_state.position():
        return False

    # Check for obstacles along the path
    from_row, from_col = from_state.position()
    to_row, to_col = to_state.position()
    line_cells = bresenham_line(from_col, from_row, to_col, to_row)

    for cell in line_cells:
        if not track.is_valid_coordinate(cell):
            return False
        if track.get_cell_type(cell) == 'O':
            return False

    cell_type = track.get_cell_type(to_state.position())
    if cell_type == 'O':
        return False
    if cell_type == 'G':
        # TODO: check if needed
        if abs(to_state.v_row) > abs(from_state.v_row) or abs(to_state.v_col) > abs(from_state.v_col):
            return False  # No acceleration on grass
    return True

def precompute_goal_heuristic(track: Track) -> np.ndarray:
    heuristic_map = np.full((track.rows, track.cols), np.inf)
    goals = track.getGoalCoordinates()
    queue = deque()

    for r, c in goals:
        heuristic_map[r, c] = 0
        queue.append((r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not track.is_valid_coordinate((nr, nc)):
                continue
            cell_type = track.get_cell_type((nr, nc))
            if cell_type == 'O':
                continue

            if heuristic_map[nr, nc] > heuristic_map[r, c] + 1:
                heuristic_map[nr, nc] = heuristic_map[r, c] + 1
                queue.append((nr, nc))

    return heuristic_map

def build_graph(track: Track, start_state: CarState, max_depth):
    g = nx.DiGraph()
    queue = deque([(start_state, 0)])
    visited = set()
    visited_states = []

    while queue:
        current, depth = queue.popleft()
        if depth > max_depth or current in visited:
            continue
        visited.add(current)
        visited_states.append(current)

        for ax in [-1, 0, 1]:
            for ay in [-1, 0, 1]:
                new_vr = current.v_row + ax
                new_vc = current.v_col + ay
                new_r = current.row + new_vr
                new_c = current.col + new_vc
                new_state = CarState(new_r, new_c, new_vr, new_vc)

                if is_valid_transition(track, current, new_state):
                    g.add_edge(current, new_state, weight=1)
                    queue.append((new_state, depth + 1))

    return g

def heuristic(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def find_goal_node(graph, goals):
    for node in graph.nodes():
        if node.position() in goals:
            return node
    return None

def reached_goal(state: CarState, goals: list[tuple[int, int]]) -> bool:
    return state.position() in goals

def find_best_local_goal(graph, goals, heuristic_map):
    best_node = None
    best_h = float('inf')

    for node in graph.nodes:
        h = heuristic_map[node.row, node.col]
        if h < best_h:
            best_h = h
            best_node = node
    return best_node

def solve_chunked_astar(track: Track, start_state: CarState, goals: list[tuple[int, int]], heuristic_map: np.ndarray, narrowness_map, max_depth, visualize):
    current_state = start_state
    full_path = [current_state]

    while not reached_goal(current_state, goals):
        graph = build_graph(track, current_state, max_depth=max_depth)

        if visualize:
            draw_graph(graph, track, title="Current Graph")
            plt.show()

        if not graph or len(graph) == 0:
            print("No further graph could be built. Aborting.")
            return full_path

        local_goal = find_best_local_goal(graph, goals, heuristic_map)
        if not local_goal:
            print("No reachable local goal found.")
            return full_path

        try:
            partial_path = nx.astar_path(
                graph,
                current_state,
                local_goal,
                heuristic=lambda n, _: combined_heuristic(n, heuristic_map, narrowness_map),
                weight='weight'
            )
        except nx.NetworkXNoPath:
            print("No path found in this chunk.")
            return full_path

        # Add to path, skip duplicate current node
        full_path += partial_path[1:]
        current_state = partial_path[-1]

    return full_path

def combined_heuristic(state: CarState, heuristic_map: np.ndarray, narrowness_map: np.ndarray, alpha=1.0, beta=1.2):
    """
    A* heuristic with:
    - alpha: weight for distance to goal (precomputed)
    - beta: penalty for narrow areas (inverted narrowness score)
    """
    row, col = state.row, state.col
    h_dist = heuristic_map[row, col]

    # Avoid division by 0 and walls
    narrow_value = narrowness_map[row, col]
    if np.isnan(narrow_value) or narrow_value == 0:
        penalty = 10  # arbitrary high penalty for walls or unknowns
    else:
        penalty = 1 / narrow_value  # narrower = bigger penalty

    # TODO: investigate penalty impact
    return 1#alpha * h_dist + beta * penalty

def save_path_as_csv(path, output_path, track):
    with open(output_path, 'w') as f:
        for state in path:
            # adjust for different origins
            transformed_row = track.rows - 1 - state.row
            f.write(f"{state.col},{transformed_row}\n")

def find_path(track_path, visualize, output, depth):
    track = Track(loadTrack(track_path))

    start = track.getStartCoordinates()
    if start is None:
        print("No start point 'S' found on the track!")
        return

    goals = track.getGoalCoordinates()
    if not goals:
        print("No goal point(s) 'F' found on the track!")
        return

    start_state = CarState(start[0], start[1], 0, 0)

    print("Precomputing heuristic map...")
    heuristic_map = precompute_goal_heuristic(track)

    if visualize:
        plt.imshow(heuristic_map, cmap='viridis', origin='upper')
        plt.colorbar(label="Distance to Goal")
        plt.title("Precomputed Heuristic Map")
        plt.show()

    narrowness_map = compute_narrowness_map(track, radius=10)

    if visualize:
        draw_narrowness_map(track, narrowness_map)

    print("Running chunked A*...")
    path = solve_chunked_astar(track, start_state, goals, heuristic_map, narrowness_map, depth, visualize=visualize)

    if visualize:
        draw_path_on_track(track=track, path=path, title="Racetrack A* Result", show_acceleration=True)

    if path:
        print(f"Path found with {len(path)} steps.")
        save_path_as_csv(path, output, track)
        run_visualization_in_docker(
            trackFilePath=track_path,
            routeFilePath=output,
            outputPdfPath="visualizations/final_output.pdf"
        )
    else:
        print("No valid path found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run construction heuristic on a racetrack file.")
    parser.add_argument(
        "--track", "-t",
        type=str,
        default="tracks/track_02.t",
        help="Path to the track file. (default: tracks/track_02.t)"
    )

    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        default=False,
        help="Enable graphical visualization of the search process."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="routes/output.csv",
        help="Path to save the output route file. (default: routes/output.csv)"
    )

    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=1,
        help="Maximum depth for graph building. (default: 1)"
    )

    args = parser.parse_args()

    find_path(
        track_path=args.track,
        visualize=args.visualize,
        output=args.output,
        depth=args.depth
    )