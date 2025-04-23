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

visited_global = set()

def compute_narrowness_map(track: Track, radius: int = 1) -> np.ndarray:
    narrowness_map = np.full((track.rows, track.cols), np.nan)

    for r in range(track.rows):
        for c in range(track.cols):
            if not track.is_valid_coordinate((r, c)):
                continue

            free = 0
            total = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if track.is_valid_coordinate((nr, nc)):
                        total += 1
                        cell = track.get_cell_type((nr, nc))
                        if cell != 'O':  # count only non-wall neighbors
                            free += 1
            if total > 0:
                narrowness_map[r, c] = free / total * 100
    valid_mask = np.isfinite(narrowness_map)

    if np.any(valid_mask):
        min_val = np.min(narrowness_map[valid_mask])
        max_val = np.max(narrowness_map[valid_mask])
        range_val = max_val - min_val

        if range_val > 0:
            narrowness_map[valid_mask] = (narrowness_map[valid_mask] - min_val) / range_val
        else:
            narrowness_map[valid_mask] = 1.0

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
        if abs(to_state.v_row) > abs(from_state.v_row) or abs(to_state.v_col) > abs(from_state.v_col):
            return False  # No acceleration on grass
    return True


def precompute_goal_heuristic(track: Track):
    distance_map = np.full((track.rows, track.cols), np.inf)
    goals = track.getGoalCoordinates()
    queue = deque()

    for r, c in goals:
        distance_map[r, c] = 0
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

            if distance_map[nr, nc] > distance_map[r, c] + 1:
                distance_map[nr, nc] = distance_map[r, c] + 1
                queue.append((nr, nc))

    finite_mask = np.isfinite(distance_map)
    if np.any(finite_mask):
        max_h = np.max(distance_map[finite_mask])
        distance_map[finite_mask] /= max_h

    return distance_map


def build_graph(track: Track, start_state: CarState, max_depth, narrowness_map, distance_map, alpha, beta, gamma):
    g = nx.DiGraph()
    queue = deque([(start_state, 0)])
    visited = set()

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth or current in visited_global:
            continue
        visited.add(current)
        visited_global.add(current)

        for ax in [-1, 0, 1]:
            for ay in [-1, 0, 1]:
                new_vr = current.v_row + ax
                new_vc = current.v_col + ay
                new_r = current.row + new_vr
                new_c = current.col + new_vc
                new_state = CarState(new_r, new_c, new_vr, new_vc)

                if is_valid_transition(track, current, new_state):
                    d = distance_map[new_r][new_c]
                    narrowness = narrowness_map[new_r][new_c]
                    narrow_penalty = 1.0 - narrowness if np.isfinite(narrowness) else 1.0
                    speed = abs(new_vr) + abs(new_vc)

                    # Speed penalty calculation
                    if narrowness >= 0.8:
                        speed_penalty = -0.05 * gamma * speed
                    elif narrowness >= 0.5:
                        speed_penalty = 0.0
                    else:
                        penalty_factor = (1.0 - narrowness) ** 2
                        speed_penalty = gamma * speed * penalty_factor

                    weight = (
                            alpha * d +
                            beta * narrow_penalty +
                            speed_penalty
                    )

                    g.add_edge(current, new_state, weight=weight)

                    if new_state not in visited:
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


def find_best_local_goal(graph, current_state, distance_map, narrowness_map, alpha, beta, gamma):
    best_node = None
    best_score = float('inf')
    best_goal_node = None
    best_goal_distance = float('inf')

    for node in graph.nodes:
        # Skip if node is the same as current to avoid self-loop
        if node == current_state:
            continue

        # Distance to goal
        d = distance_map[node.position()]
        if not np.isfinite(d):
            continue  # skip unreachable positions

        if d == 0.0:
            # It's a goal node → find the closest one to current_state
            dist_to_current = np.linalg.norm(np.array(node.position()) - np.array(current_state.position()))
            if dist_to_current < best_goal_distance:
                best_goal_distance = dist_to_current
                best_goal_node = node
            continue

        speed = abs(node.v_row) + abs(node.v_col)
        narrowness = narrowness_map[node.position()]
        narrow_penalty = 1.0 - narrowness

        if narrowness >= 0.8:
            # Only very slight reward for speed in very wide open areas
            speed_penalty = -0.05 * gamma * speed
        elif narrowness >= 0.5:
            # No reward or penalty in mid-width areas
            speed_penalty = 0.0
        else:
            # Penalize speed more steeply as narrowness decreases
            penalty_factor = (1.0 - narrowness) ** 2  # quadratic penalty
            speed_penalty = gamma * speed * penalty_factor

        score = (
                alpha * d +
                beta * narrow_penalty +
                speed_penalty
        )

        if score < best_score:
            best_score = score
            best_node = node

    return best_goal_node if best_goal_node is not None else best_node


def solve_chunked_astar(track: Track, start_state: CarState, goals: list[tuple[int, int]], distance_map: np.ndarray,
                        narrowness_map, max_depth, visualize):
    current_state = start_state
    full_path = [current_state]

    # distance
    alpha = 0.5
    # narrow
    beta = 0.0
    # speed
    gamma = 0.5

    while not reached_goal(current_state, goals):
        graph = build_graph(track, current_state, max_depth, narrowness_map, distance_map, alpha, beta, gamma)

        if visualize:
            draw_graph(graph, track, title="Current Graph")
            plt.show()

        if not graph or len(graph) == 0:
            print("No further graph could be built. Aborting.")
            return full_path

        local_goal = find_best_local_goal(graph, current_state, distance_map, narrowness_map, alpha, beta, gamma)
        if not local_goal:
            print("No reachable local goal found.")
            return full_path

        try:
            partial_path = nx.astar_path(
                graph,
                current_state,
                local_goal,
                heuristic=lambda n, _: combined_heuristic(n, distance_map, narrowness_map, alpha, beta, gamma),
                weight='weight'
            )
        except nx.NetworkXNoPath:
            print("No path found in this chunk.")
            return full_path

        # Add to path, skip duplicate current node
        full_path += partial_path[1:]
        current_state = partial_path[-1]

    return full_path


def combined_heuristic(
        state: CarState,
        distance_map: np.ndarray,
        narrowness_map: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float
):
    row, col = state.row, state.col
    dist = distance_map[row, col]

    # Narrowness penalty: higher when space is tight
    narrow = narrowness_map[row, col]
    narrow_penalty = 1 - narrow  # already normalized between 0–1

    # Speed penalty: discourage high speed in general or tune it with narrowness
    speed = abs(state.v_row) + abs(state.v_col)

    beta_scaled = beta * dist # less narrowness penalty near to goal TODO: tune this

    return alpha * dist + beta_scaled * narrow_penalty + gamma * speed


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
    distance_map = precompute_goal_heuristic(track)

    if visualize:
        plt.imshow(distance_map, cmap='viridis', origin='upper')
        plt.colorbar(label="Distance to Goal")
        plt.title("Precomputed Heuristic Map")
        plt.show()

    narrowness_map = compute_narrowness_map(track, radius=5)

    if visualize:
        draw_narrowness_map(track, narrowness_map)

    print("Running chunked A*...")
    path = solve_chunked_astar(track, start_state, goals, distance_map, narrowness_map, depth, visualize)

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
