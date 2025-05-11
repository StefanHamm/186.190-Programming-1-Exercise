import pyperf

from src.helper import Track, loadTrack
from src.bfs import bfs_racetrack
from src.construction import solve_chunked_astar, precompute_goal_heuristic, compute_narrowness_map
from src.state import CarState

def benchmark_bfs():
    bfs_racetrack(track)

def benchmark_construction():
    solve_chunked_astar(track, start_state, goals, distance_map, narrowness_map, depth, visualize=False, parameters=None)

if __name__ == "__main__":
    track = Track(loadTrack("tracks/track_02.t"))
    start = track.getStartCoordinates()
    start_state = CarState(start[0], start[1], 0, 0)
    goals = track.getGoalCoordinates()
    distance_map = precompute_goal_heuristic(track)
    narrowness_map = compute_narrowness_map(track)
    depth = 3

    runner = pyperf.Runner()
    runner.bench_func("bfs", benchmark_bfs)
    runner.bench_func("construction", benchmark_construction)
