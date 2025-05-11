import os
import tempfile
import shutil
import atexit

# --- START FIX ---
# Create a temporary directory for matplotlib's config
# This needs to happen BEFORE matplotlib is imported by any of your modules.
# The temp directory will be automatically cleaned up on program exit.
try:
    # Create a temporary directory for matplotlib's config
    mpl_config_dir = tempfile.mkdtemp(prefix="mpl_config_")
    os.environ['MPLCONFIGDIR'] = mpl_config_dir

    # Ensure the temporary directory is cleaned up when the script exits
    def cleanup_mpl_config():
        if os.path.exists(mpl_config_dir):
            shutil.rmtree(mpl_config_dir)
            # print(f"Cleaned up MPLCONFIGDIR: {mpl_config_dir}") # Optional: for debugging

    atexit.register(cleanup_mpl_config)
except Exception as e:
    print(f"Warning: Could not set up temporary MPLCONFIGDIR: {e}")
# --- END FIX ---

import pyperf

# Your original imports - matplotlib will now use MPLCONFIGDIR
from src.helper import Track, loadTrack
from src.bfs import bfs_racetrack # This will import src.construction, which imports matplotlib
from src.construction import solve_chunked_astar, precompute_goal_heuristic, compute_narrowness_map
from src.state import CarState

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

if __name__ == "__main__":
    # Define globals here. When pyperf spawns a worker, it re-runs this script,
    # so these will be available in the worker's global scope for the benchmark functions.
    track = Track(loadTrack("tracks/track_02.t"))
    start = track.getStartCoordinates()
    start_state = CarState(start[0], start[1], 0, 0)
    goals = track.getGoalCoordinates()
    distance_map = precompute_goal_heuristic(track)
    narrowness_map = compute_narrowness_map(track)
    depth = 3

    runner = pyperf.Runner()
    # The functions benchmark_bfs and benchmark_construction will be called
    # by pyperf, and they will pick up the global variables defined above.
    #runner.bench_func("bfs", benchmark_bfs)
    runner.bench_func("construction", benchmark_construction)