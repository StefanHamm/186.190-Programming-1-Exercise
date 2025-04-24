from collections import deque
import heapq
from itertools import count

from src.construction import save_path_as_csv
from src.helper import Track, loadTrack, bresenham_line, run_visualization_in_docker
from src.state import CarState
import argparse
import networkx as nx

def manhattan_heuristic(state: CarState, goal_positions: set[tuple[int, int]]) -> int:
    return min(abs(state.row - gr) + abs(state.col - gc) for gr, gc in goal_positions)

def a_star_racetrack(track: Track) -> list[CarState]:
    start_pos = track.getStartCoordinates()
    if start_pos is None:
        print("No start found.")
        return []

    goal_positions = set(track.getGoalCoordinates())
    start_state = CarState(*start_pos, 0, 0)

    open_heap = []  # priority queue as min-heap: (f_cost, count, g_cost, CarState, path)
    counter = count()
    heapq.heappush(open_heap, (0, next(counter), 0, start_state, []))
    visited = set()

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
                    track.get_cell_type((r, c)) in ['O', None, 'G']  # TODO: allow grass
                    for r, c in cells_crossed
                ):
                    continue

                if new_state not in visited:
                    new_g = g + 1
                    h = manhattan_heuristic(new_state, goal_positions)
                    new_f = new_g + h
                    heapq.heappush(open_heap, (new_f, next(counter), new_g, new_state, new_path))

    print("No valid path found.")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run construction heuristic on a racetrack file.")
    parser.add_argument(
        "--track", "-t",
        type=str,
        default="tracks/track_02.t",
        help="Path to the track file. (default: tracks/track_02.t)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="routes/output.csv",
        help="Path to save the output route file. (default: routes/output.csv)"
    )

    args = parser.parse_args()

    print("\n--- Running A* Racetrack Algorithm ---")
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