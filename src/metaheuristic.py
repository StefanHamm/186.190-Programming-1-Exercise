from typing import List
import numpy as np

from src.bfs import bfs_racetrack
from src.state import CarState
from src.helper import Track, loadTrack


def evaluate_path_cost(path: List[CarState]) -> float:
    if not path:
        return float('inf')
    return len(path)


def mutate_params(a: float, b: float, iteration: int, max_iterations: int, initial_scale: float = 0.1) -> (float, float):
    scale = initial_scale * (0.9 ** (iteration / max_iterations))

    a_mutated = max(1, int(a + np.random.randint(-2, 3)))
    b_mutated = b + np.random.normal(0, scale * 2)

    return a_mutated, b_mutated

def optimize_brush_params(track: Track, path_file: str, initial_a: float, initial_b: float, iterations: int = 50) -> (float, float, float):
    best_a, best_b = initial_a, initial_b
    best_cost = float('inf')

    for i in range(iterations):
        a, b = mutate_params(best_a, best_b, i, iterations)

        print(f"Iteration {i}: Testing a={a}, b={b}")

        brushed_track = track.getBrushedTrack(path_file, int(a), True, b)
        path = bfs_racetrack(brushed_track)
        cost = evaluate_path_cost(path)

        if cost < best_cost:
            best_cost = cost
            best_a, best_b = a, b
            print(f"Iteration {i}: New best cost {best_cost} with a={best_a}, b={best_b}")

    print(f"Optimization complete: Best cost {best_cost} with a={best_a}, b={best_b}")
    return best_a, best_b, best_cost

if __name__ == "__main__":
    # Example usage
    track = Track(loadTrack("tracks/track_03.t"))
    path_file = "benchmark/path/construction_path_track_03.t.csv"  # Path to the path file
    initial_a = 2.0
    initial_b = 1.0

    best_a, best_b, best_cost = optimize_brush_params(track, path_file, initial_a, initial_b)
    print(f"Optimized parameters: a={best_a}, b={best_b}, cost={best_cost}")