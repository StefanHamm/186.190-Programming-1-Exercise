import csv
import os
import matplotlib.pyplot as plt

# File path
csv_path = 'benchmark/metaheuristic_bfs.txt'

# Prepare lists to store data
tracks = []
bfs_times = []
meta_times = []
bfs_mems = []
meta_mems = []

# Read the CSV data manually
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        tracks.append(row['track'].replace('.t', ''))
        bfs_times.append(float(row['bfs_time']))
        meta_times.append(float(row['meta_time']))
        bfs_mems.append(float(row['bfs_mem']))
        meta_mems.append(float(row['meta_mem']))

# Compute improvement factors
time_factors = [b / m if m > 0 else 0 for b, m in zip(bfs_times, meta_times)]
memory_factors = [b / m if m > 0 else 0 for b, m in zip(bfs_mems, meta_mems)]

# Ensure output directory exists
os.makedirs('benchmark/plots', exist_ok=True)

# Plot Time Improvement
plt.figure(figsize=(10, 6))
plt.bar(tracks, time_factors)
plt.ylabel('Time Improvement Factor (BFS / Metaheuristic) in seconds')
plt.title('Time Improvement Per Track')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('benchmark/plots/meta_bfs_time_improvement.png')
plt.close()

# Plot Memory Improvement
plt.figure(figsize=(10, 6))
plt.bar(tracks, memory_factors, color='orange')
plt.ylabel('Memory Improvement Factor (BFS / Metaheuristic) in MB')
plt.title('Memory Improvement Per Track')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('benchmark/plots/meta_bfs_memory_improvement.png')
plt.close()