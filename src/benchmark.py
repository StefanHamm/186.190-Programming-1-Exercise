import os

import pyperf

# Your original imports - matplotlib will now use MPLCONFIGDIR
from src.helper import Track, loadTrack
from src.bfs import bfs_racetrack # This will import src.construction, which imports matplotlib
from src.construction import solve_chunked_astar, precompute_goal_heuristic, compute_narrowness_map
from src.state import CarState
from memory_profiler import memory_usage
from enum import Enum
import argparse

class BenchmarkTarget(Enum):
    BFS = "bfs"
    CONSTRUCTION = "construction"

    def __str__(self):
        return self.value

# Global variables for benchmark functions - define them before the functions
# if they are needed at function definition time (not strictly here, but good practice)
# However, these are actually only needed when the __main__ block runs,
# so they will be defined before benchmark_bfs and benchmark_construction are *called* by pyperf.
# For clarity, let's keep them in __main__ as pyperf will re-execute the script for workers.
# track = None
# start_state = None
# goals = None
# distance_map = None
# narrowness_map = None
# depth = None

def benchmark_bfs():
    # These globals will be defined in the __main__ block when pyperf runs the worker
    bfs_racetrack(track)

def benchmark_construction():
    # These globals will be defined in the __main__ block when pyperf runs the worker
    solve_chunked_astar(track, start_state, goals, distance_map, narrowness_map, depth, visualize=False, parameters=None)

def profile_memory(target: BenchmarkTarget, track_name: str):
    # Memory profiling
    memory_usage_file_name = f"benchmark/{target}_memory_{track_name}.json"

    if target == BenchmarkTarget.BFS:
        mem_usage = memory_usage((bfs_racetrack, (track,)), interval=0.1)
    elif target == BenchmarkTarget.CONSTRUCTION:
        mem_usage = memory_usage((solve_chunked_astar, (track, start_state, goals, distance_map, narrowness_map, depth, False, None,)), interval=0.1)

    with open(memory_usage_file_name, "w") as f:
        for entry in mem_usage:
            f.write(f"{entry}\n")

def parse_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--target",
        type=str,
        choices=["bfs", "construction"],
        default='construction',
        help="Which benchmark to run"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth parameter for construction benchmark (only used when target=construction"
    )

    parser.add_argument(
        "--track",
        type=str,
        default="track_02.t",
        help="Track file name inside 'tracks' directory (default: track_02.t)"
    )

    return parser.parse_args()

# parse custom args to worker (child) processes of perf
def add_custom_args(cmd, args):
    if args.track:
        cmd.extend(["--track", args.track])
    if args.target:
        cmd.extend(["--target", args.target])
    if args.depth:
        cmd.extend(["--depth", str(args.depth)])

if __name__ == "__main__":
    runner = pyperf.Runner(add_cmdline_args=add_custom_args)

    parser = runner.argparser

    args = parse_args(parser)

    track_name = args.track
    target = BenchmarkTarget(args.target)

    # Define globals here. When pyperf spawns a worker, it re-runs this script,
    # so these will be available in the worker's global scope for the benchmark functions.
    track = Track(loadTrack(f"tracks/{track_name}"))

    if target == BenchmarkTarget.BFS:
        bench = runner.bench_func("bfs", benchmark_bfs)
    elif target == BenchmarkTarget.CONSTRUCTION:
        start = track.getStartCoordinates()
        start_state = CarState(start[0], start[1], 0, 0)
        goals = track.getGoalCoordinates()
        distance_map = precompute_goal_heuristic(track)
        narrowness_map = compute_narrowness_map(track)
        depth = args.depth

        bench = runner.bench_func("construction", benchmark_construction)
    else:
        raise ValueError(f"{target} is not a valid benchmark target")

    if runner.args.worker:
        # This is a worker process spawned by pyperf, dont run memory measurement and dont dump results yet
        exit(0)

    perf_dump_file_name = f"benchmark/{target}_{track_name}.json"

    if os.path.exists(perf_dump_file_name):
        os.remove(perf_dump_file_name)

    bench.dump(perf_dump_file_name)

    profile_memory(target, track_name)
