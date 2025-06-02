import argparse
import os.path
from typing import List
import numpy as np

from src.bfs import bfs_racetrack
from src.construction import save_path_as_csv
from src.state import CarState
from src.helper import Track, loadTrack, run_visualization_in_docker


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
    best_path = []

    for i in range(iterations):
        a, b = mutate_params(best_a, best_b, i, iterations)

        print(f"Iteration {i}: Testing a={a}, b={b}")

        brushed_track = track.getBrushedTrack(path_file, int(a), True, b)
        path = bfs_racetrack(brushed_track)
        cost = evaluate_path_cost(path)

        if cost < best_cost:
            best_path = path
            best_cost = cost
            best_a, best_b = a, b
            print(f"Iteration {i}: New best cost {best_cost} with a={best_a}, b={best_b}")

    print(f"Optimization complete: Best cost {best_cost} with a={best_a}, b={best_b}")
    return best_a, best_b, best_cost, best_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track", "-t",
        type=str,
        default="track_03.t",
    )

    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    # Example usage
    track = Track(loadTrack(f"tracks/{args.track}"))
    path_file = f"benchmark/path/construction_path_{args.track}.csv"  # Path to the path file
    initial_a = 2.0
    initial_b = 1.0

    best_a, best_b, best_cost, best_path = optimize_brush_params(track, path_file, initial_a, initial_b, args.iterations)
    print(f"Optimized parameters: a={best_a}, b={best_b}, cost={best_cost}")

    best_params_file_name = f"benchmark/metaheuristic/params_{args.track}.csv"

    if os.path.exists(best_params_file_name):
        os.remove(best_params_file_name)

    with open(best_params_file_name, 'w') as f:
        f.write(f"a,{best_a}\n")
        f.write(f"b,{best_b}\n")
        f.write(f"cost,{best_cost}\n")

    brushed_track = track.getBrushedTrack(path_file, best_a, True, best_b)

    brushed_track_file_name = f"benchmark/metaheuristic/brushed_{args.track}.t"

    if os.path.exists(brushed_track_file_name):
        os.remove(brushed_track_file_name)

    brushed_track.saveToFile(brushed_track_file_name)

    best_path_file_name = f"benchmark/metaheuristic/path_{args.track}.csv"

    if os.path.exists(best_path_file_name):
        os.remove(best_path_file_name)

    save_path_as_csv(best_path, best_path_file_name, brushed_track)

    run_visualization_in_docker(
        trackFilePath=brushed_track_file_name,
        routeFilePath=best_path_file_name,
        outputPdfPath=f"benchmark/metaheuristic/{args.track}.pdf"
    )