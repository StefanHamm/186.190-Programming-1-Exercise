from collections import deque
import heapq
from itertools import count

from src.construction import save_path_as_csv
from src.helper import Track, loadTrack, bresenham_line, run_visualization_in_docker, normalize_map
from src.state import CarState
import argparse
import numpy as np

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
                for nr in range(max(0, r-20), min(rows, r+20))
                for nc in range(max(0, c-20), min(cols, c+20))
                if track.is_valid_coordinate((nr, nc)) and track.get_cell_type((nr, nc)) == 'O'
            ) if any(track.get_cell_type((nr, nc)) == 'O' for nr in range(max(0, r-20), min(rows, r+20)) for nc in range(max(0, c-20), min(cols, c+20))) else 0

            safe_speed_map[r, c] = np.sqrt(2 * d) if d > 0 else 0
    safe_speed_map = normalize_map(safe_speed_map)

    return distance_map, narrowness_map, safe_speed_map

def combined_heuristic(state: CarState, distance_map, narrowness_map, safe_speed_map, alpha=1.0, beta=1.0, gamma=0.3, delta=1.0):
    row, col = state.row, state.col
    if not (0 <= row < distance_map.shape[0] and 0 <= col < distance_map.shape[1]):
        return float('inf')

    distance_score = distance_map[row, col]
    narrowness_score = 1.0 - narrowness_map[row, col]
    safe_speed = safe_speed_map[row, col]
    speed = np.hypot(state.v_row, state.v_col)

    overspeed_penalty = (speed - safe_speed)**2 if speed > safe_speed else 0

    return alpha * distance_score + beta * narrowness_score + gamma * speed + delta * overspeed_penalty

def a_star_racetrack(track: Track) -> list[CarState]:
    start_pos = track.getStartCoordinates()
    if start_pos is None:
        print("No start found.")
        return []

    goal_positions = set(track.getGoalCoordinates())
    start_state = CarState(*start_pos, 0, 0)

    distance_map, narrowness_map, safe_speed_map = compute_maps(track)

    open_heap = []
    counter = count()
    heapq.heappush(open_heap, (0, next(counter), 0, start_state, []))
    visited = set()

    alpha, beta, gamma, delta = 1.0, 1.0, 0.3, 2.0  # (tune later)

    while open_heap:
        f, _, g, current_state, path = heapq.heappop(open_heap)
        if current_state in visited:
            continue
        visited.add(current_state)

        new_path = path + [current_state]

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

                cells_crossed = bresenham_line(current_state.col, current_state.row, new_col, new_row)
                if any(
                    not track.is_valid_coordinate((r, c)) or
                    track.get_cell_type((r, c)) in ['O', None, 'G']
                    for r, c in cells_crossed
                ):
                    continue

                if new_state not in visited:
                    new_g = g + 1
                    h = combined_heuristic(new_state, distance_map, narrowness_map, safe_speed_map, alpha, beta, gamma, delta)
                    new_f = new_g + h
                    heapq.heappush(open_heap, (new_f, next(counter), new_g, new_state, new_path))

    print("No valid path found.")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run global A* with precomputed heuristics.")
    parser.add_argument("--track", "-t", type=str, default="tracks/track_02.t")
    parser.add_argument("--output", "-o", type=str, default="routes/output.csv")
    args = parser.parse_args()

    print("\n--- Running Global Heuristic A* Racetrack ---")
    track = Track(loadTrack(args.track))
    path_states = a_star_racetrack(track)

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