import os
import json
import matplotlib.pyplot as plt

def load_mean_times(track_ids, base_path="benchmark"):
    bfs_means = []
    construction_means = []
    labels = []

    for i in track_ids:
        track_label = f"Track {i:02d}"
        labels.append(track_label)

        bfs_path = os.path.join(base_path, f"bfs_track_{i:02d}.t.json")
        cons_path = os.path.join(base_path, f"construction_track_{i:02d}.t.json")

        def get_mean(filepath):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    return data["metadata"].get("mean")
            except (FileNotFoundError, json.JSONDecodeError):
                return None

        bfs_means.append(get_mean(bfs_path))
        construction_means.append(get_mean(cons_path))

    return labels, bfs_means, construction_means

def load_memory_usage(track_ids, base_path="benchmark"):
    bfs_memory = []
    construction_memory = []
    labels = []

    for i in track_ids:
        track_label = f"Track {i:02d}"
        labels.append(track_label)

        bfs_path = os.path.join(base_path, f"bfs_track_{i:02d}.t.json")
        cons_path = os.path.join(base_path, f"construction_track_{i:02d}.t.json")

        def get_memory_usage(filepath):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    return data["metadata"]["memory"].get("usage")
            except (FileNotFoundError, json.JSONDecodeError, AttributeError):
                return None

        bfs_memory.append(get_memory_usage(bfs_path))
        construction_memory.append(get_memory_usage(cons_path))

    return labels, bfs_memory, construction_memory

def plot_comparison(labels, bfs_means, construction_means, output_path):
    x = list(range(len(labels)))
    bar_width = 0.35

    bfs_clean = [v if v is not None else 0 for v in bfs_means]
    cons_clean = [v if v is not None else 0 for v in construction_means]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - bar_width / 2 for i in x], bfs_clean, width=bar_width, label="BFS", color="skyblue")
    ax.bar([i + bar_width / 2 for i in x], cons_clean, width=bar_width, label="Construction", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title("BFS vs Construction Mean Runtime per Track")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for i, (bm, cm) in enumerate(zip(bfs_means, construction_means)):
        if bm is None:
            ax.text(i - bar_width / 2, 0.01, "X", ha="center", va="bottom", color="red")
        if cm is None:
            ax.text(i + bar_width / 2, 0.01, "X", ha="center", va="bottom", color="red")

    plt.tight_layout()

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

    plt.close()

def plot_memory_comparison(labels, bfs_mem, cons_mem, output_path):
    x = list(range(len(labels)))
    bar_width = 0.35

    bfs_clean = [v if v is not None else 0 for v in bfs_mem]
    cons_clean = [v if v is not None else 0 for v in cons_mem]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar([i - bar_width / 2 for i in x], bfs_clean, width=bar_width, label="BFS", color="skyblue")
    ax.bar([i + bar_width / 2 for i in x], cons_clean, width=bar_width, label="Construction", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("BFS vs Construction Memory Usage per Track")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for i, (bm, cm) in enumerate(zip(bfs_mem, cons_mem)):
        if bm is None:
            ax.text(i - bar_width / 2, 0.01, "X", ha="center", va="bottom", color="red")
        if cm is None:
            ax.text(i + bar_width / 2, 0.01, "X", ha="center", va="bottom", color="red")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Memory plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    tracks = range(2, 11)
    labels, bfs, construction = load_mean_times(tracks)
    plot_comparison(labels, bfs, construction, output_path="benchmark/plots/runtime_comparison.png")

    labels, bfs_mem, construction_mem = load_memory_usage(tracks)
    plot_memory_comparison(labels, bfs_mem, construction_mem, output_path="benchmark/plots/memory_comparison.png")