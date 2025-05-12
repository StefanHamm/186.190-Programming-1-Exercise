from collections import deque

from src.construction import save_path_as_csv
from src.helper import Track, loadTrack, run_visualization_in_docker, is_invalid_move
from src.state import CarState
import argparse


def bfs_racetrack(track: Track) -> list[CarState]:
    start_pos = track.getStartCoordinates()
    if start_pos is None:
        print("No start found.")
        return []

    goal_positions = set(track.getGoalCoordinates())
    start_state = CarState(*start_pos, 0, 0)

    visited = set()
    queue = deque([(start_state, [])])  # (CarState, path)

    while queue:
        current_state, path = queue.popleft()
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

                # Skip zero movement (no velocity)
                if new_vx == 0 and new_vy == 0:
                    continue

                new_row = current_state.row + new_vx
                new_col = current_state.col + new_vy
                new_state = CarState(new_row, new_col, new_vx, new_vy)

                if new_state in visited:
                    continue

                if is_invalid_move(track, current_state, new_state):
                    continue

                queue.append((new_state, new_path))

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

    print("\n--- Running BFS Racetrack Algorithm ---")
    track = Track(loadTrack(args.track))

    path_states = bfs_racetrack(track)

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
