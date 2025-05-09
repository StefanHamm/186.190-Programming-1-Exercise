import heapq
from itertools import count
from .construction import save_path_as_csv
from .helper import Track, loadTrack, bresenham_line, run_visualization_in_docker
from .state import CarState
import argparse

def dijkstra_racetrack(track: Track) -> list[CarState]:
    start_pos = track.getStartCoordinates()
    if start_pos is None:
        print("No start found.")
        return []

    goal_positions = set(track.getGoalCoordinates())
    start_state = CarState(*start_pos, 0, 0)

    open_heap = []
    counter = count()
    heapq.heappush(open_heap, (0, next(counter), start_state, []))
    visited = set()

    while open_heap:
        g, _, current_state, path = heapq.heappop(open_heap)
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
                    heapq.heappush(open_heap, (g + 1, next(counter), new_state, new_path))

    print("No valid path found.")
    return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dijkstra's algorithm on a racetrack file.")
    parser.add_argument("--track", "-t", type=str, default="tracks/track_02.t", help="Path to the track file.")
    parser.add_argument("--output", "-o", type=str, default="routes/dijkstra_output.csv", help="Path to save the output route file.")

    args = parser.parse_args()

    print("\n--- Running Dijkstra Racetrack Algorithm ---")
    track = Track(loadTrack(args.track))

    path_states = dijkstra_racetrack(track)

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