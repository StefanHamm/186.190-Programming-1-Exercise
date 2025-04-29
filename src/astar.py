from collections import deque
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from matplotlib import pyplot as plt

from src.construction import save_path_as_csv
from src.helper import Track, loadTrack, bresenham_line, run_visualization_in_docker, normalize_map
from src.state import CarState
import argparse
import numpy as np
from src.visualizer import Visualizer
import tqdm


def compute_maps(track: Track):
    rows, cols = track.rows, track.cols

    # 1. Distance to goal (simple BFS)
    distance_map = np.full((rows, cols), np.inf)
    queue = deque(track.getGoalCoordinates())
    for r, c in track.getGoalCoordinates():
        distance_map[r, c] = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if track.is_valid_coordinate((nr, nc)) and track.get_cell_type((nr, nc)) != 'O':
                if distance_map[nr, nc] > distance_map[r, c] + 1:
                    distance_map[nr, nc] = distance_map[r, c] + 1
                    queue.append((nr, nc))

    distance_map = normalize_map(distance_map)

    # 2. Narrowness map
    narrowness_map = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            if not track.is_valid_coordinate((r, c)):
                continue
            free = 0
            total = 0
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = r + dr, c + dc
                    if track.is_valid_coordinate((nr, nc)):
                        total += 1
                        if track.get_cell_type((nr, nc)) != 'O':
                            free += 1
            if total > 0:
                narrowness_map[r, c] = free / total
    narrowness_map = normalize_map(narrowness_map)

    # 3. Safe speed map
    safe_speed_map = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            if not track.is_valid_coordinate((r, c)):
                continue
            d = min(
                np.hypot(r - nr, c - nc)
                for nr in range(max(0, r - 20), min(rows, r + 20))
                for nc in range(max(0, c - 20), min(cols, c + 20))
                if track.is_valid_coordinate((nr, nc)) and track.get_cell_type((nr, nc)) == 'O'
            ) if any(track.get_cell_type((nr, nc)) == 'O' for nr in range(max(0, r - 20), min(rows, r + 20)) for nc in
                     range(max(0, c - 20), min(cols, c + 20))) else 0

            safe_speed_map[r, c] = np.sqrt(2 * d) if d > 0 else 0

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(distance_map, cmap='viridis', origin='upper')
    plt.title("Distance to Goal")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(narrowness_map, cmap='coolwarm_r', origin='upper')
    plt.title("Narrowness Map")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(safe_speed_map, cmap='plasma', origin='upper')
    plt.title("Safe Speed Map")
    plt.colorbar()

    plt.suptitle("Precomputed Maps")
    plt.tight_layout()
    plt.show()

    return distance_map, narrowness_map, safe_speed_map


def combined_heuristic(state: CarState, distance_map, narrowness_map, safe_speed_map, alpha, beta, gamma,
                       delta):
    row, col = state.row, state.col
    if not (0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]):
        return float('inf')

    distance_score = distance_map[row, col]
    narrowness_score = 1.0 - narrowness_map[row, col]
    safe_speed = safe_speed_map[row, col]
    speed = np.hypot(state.v_row, state.v_col)

    overspeed_penalty = (speed - safe_speed) ** 2 if speed > safe_speed else 0

    return alpha * distance_score + beta * narrowness_score + gamma * speed + delta * overspeed_penalty


def a_star_racetrack(track: Track,
                     distance_map,
                     narrowness_map,
                     safe_speed_map,
                     alpha,
                     beta,
                     gamma,
                     delta,
                     visualize=False) -> list[CarState]:
    start_pos = track.getStartCoordinates()
    if start_pos is None:
        print("No start found.")
        return []

    goal_positions = set(track.getGoalCoordinates())
    start_state = CarState(*start_pos, 0, 0)

    open_heap = []
    counter = count()
    heapq.heappush(open_heap, (0, next(counter), 0, start_state, []))
    visited = set()

    while open_heap:
        f, _, g, current_state, path = heapq.heappop(open_heap)

        if current_state in visited:
            continue

        visited.add(current_state)

        new_path = path + [current_state]

        if visualize:
            visualizer.draw(visited, open_heap, new_path)

        if current_state.position() in goal_positions:
            return new_path

        for dvx in [-1, 0, 1]:
            for dvy in [-1, 0, 1]:
                new_vx = current_state.v_row + dvx
                new_vy = current_state.v_col + dvy

                if new_vx == 0 and new_vy == 0:
                    continue

                new_row = current_state.row + new_vx
                new_col = current_state.col + new_vy
                new_state = CarState(new_row, new_col, new_vx, new_vy)

                if not track.is_valid_coordinate((new_row, new_col)):
                    continue

                if is_invalid_move(track, current_state, new_state):
                    continue

                if new_state not in visited:
                    new_g = g + 1
                    h = combined_heuristic(new_state, distance_map, narrowness_map, safe_speed_map, alpha, beta, gamma,
                                           delta)
                    new_f = new_g + h
                    heapq.heappush(open_heap, (new_f, next(counter), new_g, new_state, new_path))

    print("No valid path found.")
    return []


def is_invalid_move(track: Track, from_state: CarState, to_state: CarState) -> bool:
    """
    Checks if the movement from 'from_state' to 'to_state' crosses an obstacle.
    """

    # Check start or end inside an obstacle
    for check_state in [from_state, to_state]:
        row, col = check_state.row, check_state.col
        if not track.is_valid_coordinate((row, col)):
            return True
        if track.get_cell_type((row, col)) in ['O', 'G']:
            return True

    # Check line crossing
    cells_crossed = bresenham_line(from_state.col, from_state.row, to_state.col, to_state.row)
    for r, c in cells_crossed:
        if not track.is_valid_coordinate((r, c)):
            return True
        if track.get_cell_type((r, c)) == 'O':
            return True

    return False


def tune_parameters(track, distance_map, narrowness_map, safe_speed_map, tune_steps=4, parallel=False):
    best_score = float('inf')
    best_params = None

    # Grid search spaces
    alphas = np.linspace(0.5, 2.0, tune_steps)
    betas = np.linspace(0.5, 2.0, tune_steps)
    gammas = np.linspace(0.0, 1.0, tune_steps)
    deltas = np.linspace(0.0, 3.0, tune_steps)

    param_combinations = [(alpha, beta, gamma, delta)
                          for alpha in alphas
                          for beta in betas
                          for gamma in gammas
                          for delta in deltas]

    print(f"Total parameter combinations: {len(param_combinations)}")

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(
                try_parameters,
                (params, track, distance_map, narrowness_map, safe_speed_map)
            )
                for params in param_combinations
            ]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Tuning"):
                score, params = future.result()
                if score < best_score:
                    best_score = score
                    best_params = params
    else:
        for params in tqdm.tqdm(param_combinations, desc="Tuning"):
            score, params = try_parameters(
                (params, track, distance_map, narrowness_map, safe_speed_map)
            )
            if score < best_score:
                best_score = score
                best_params = params

    if best_params:
        print(
            f"Best parameters found: alpha={best_params[0]}, beta={best_params[1]}, gamma={best_params[2]}, delta={best_params[3]}")
    else:
        print("No valid parameters found.")

    return best_params


def try_parameters(args):
    params, track, distance_map, narrowness_map, safe_speed_map = args
    alpha, beta, gamma, delta = params
    try:
        path = a_star_racetrack(
            track=track,
            distance_map=distance_map,
            narrowness_map=narrowness_map,
            safe_speed_map=safe_speed_map,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            visualize=False
        )
        if path:
            return len(path), params
        else:
            return float('inf'), params
    except Exception as e:
        print(f"Exception for {params}: {e}")
        return float('inf'), params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run global A* with precomputed heuristics.")
    parser.add_argument("--track", "-t", type=str, default="tracks/track_02.t")
    parser.add_argument("--output", "-o", type=str, default="routes/output.csv")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the pathfinding process.")
    parser.add_argument("--tune", "-tune", action="store_true", help="Tune the pathfinding process.")
    parser.add_argument("--tune-steps", type=int, default=4,
                        help="Number of steps to sample per parameter axis for tuning. (default: 4)")
    parser.add_argument("--tune-parallel", action="store_true", default=False,
                        help="Use parallel processing during tuning.")
    args = parser.parse_args()

    print("\n--- Running Global Heuristic A* Racetrack ---")
    track = Track(loadTrack(args.track))

    if args.visualize:
        visualizer = Visualizer(track)

    distance_map, narrowness_map, safe_speed_map = compute_maps(track)
    # alpha, beta, gamma, delta = 1.0, 1.0, 0.3, 2.0
    alpha, beta, gamma, delta = 0.5, 0.5, 0.0, 2.0

    if args.tune:
        print("Tuning parameters...")
        best_params = tune_parameters(track, distance_map, narrowness_map, safe_speed_map, args.tune_steps, args.tune_parallel)
        if best_params:
            alpha, beta, gamma, delta = best_params
        else:
            print("No valid parameters found.")
            exit(1)

    path_states = a_star_racetrack(track, distance_map, narrowness_map, safe_speed_map, alpha, beta, gamma, delta,
                                   visualize=args.visualize)

    if path_states:
        print(f"Found path with {len(path_states)} steps.")
        save_path_as_csv(path_states, args.output, track)
        run_visualization_in_docker(
            trackFilePath=args.track,
            routeFilePath=args.output,
            outputPdfPath="visualizations/final_output.pdf"
        )
    else:
        print("No valid path found.")
