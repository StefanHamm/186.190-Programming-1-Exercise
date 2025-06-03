import argparse
import time
import os
from memory_profiler import memory_usage

from src.helper import loadTrack, Track
from src.bfs import bfs_racetrack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track", "-t",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    track = Track(loadTrack(f"benchmark/metaheuristic/brushed_{args.track}.t"))

    start_time = time.perf_counter()
    bfs_racetrack(track)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    mem_usage = memory_usage((bfs_racetrack, (track,)), interval =0.1)
    max_mem = max(mem_usage)
    min_mem = min(mem_usage)
    memory = max_mem - min_mem

    print(f"Time taken to run BFS on the track {args.track}: {elapsed_time:.4f} seconds with peak memory usage {memory:.2f} MiB")

    file_name = f"benchmark/metaheuristic/measurement_{args.track}.txt"

    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, "w") as f:
        f.write(f"Time taken: {elapsed_time:.4f} seconds\n")
        f.write(f"Peak memory usage: {memory:.2f} MiB\n")