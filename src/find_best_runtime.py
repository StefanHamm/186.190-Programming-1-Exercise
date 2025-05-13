import json

def find_best_runtime(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    best_time = float('inf')
    best_run_index = -1
    all_times = []

    for i, run in enumerate(data["benchmarks"][0]["runs"]):
        values = run.get("values", [])
        for val in values:
            all_times.append((val, i))
            if val < best_time:
                best_time = val
                best_run_index = i

    print(f"Best runtime: {best_time:.6f} seconds (Run #{best_run_index})")

# PYTHONPATH=. python -m src.find_best_runtime
if __name__ == "__main__":
    find_best_runtime("benchmark/bfs_track_02.t.json")  # Update path if needed