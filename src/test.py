import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm # For progress bars, install with: pip install tqdm
import pandas as pd # For easier data handling and analysis, install with: pip install pandas
import os
# Attempt to import the user's module
# If this script is in the same directory as the 'src' folder, this should work:
try:
    from src.construction import find_path, PathFindingStatus
except ImportError:
    # Fallback if the import path is different or for testing purposes
    print("Warning: Could not import from 'src.construction'. Using mock implementation.")
    # Mock implementation for PathFindingStatus
    # class PathFindingStatus:
    #     SUCCESSFUL = "SUCCESSFUL"
    #     FAILED_TIMEOUT = "FAILED_TIMEOUT" # Example other status
    #     FAILED_OBSTACLE = "FAILED_OBSTACLE" # Example other status

    # Mock implementation for find_path
    # def find_path(track_path, visualize, output, depth, parameters=None):
    #     """
    #     Mock function for find_path.
    #     Simulates success based on simple parameter thresholds.
    #     """
    #     alpha, beta, gamma = parameters
    #     # print(f"Mocking find_path for: {track_path}, depth={depth}, alpha={alpha:.2f}, beta={beta}, gamma={gamma:.2f}")
        
    #     # Example logic: higher alpha/gamma and depth might be better
    #     if alpha > 0.5 and gamma > 0.5 and depth >= 5:
    #         if "track_01" in track_path or "track_02" in track_path : # Succeed on specific tracks
    #              return PathFindingStatus.SUCCESSFUL
    #     if alpha > 2 and gamma > 2 and depth >= 3: # Different condition for other tracks
    #         if "track_03" in track_path or "track_04" in track_path or "track_05" in track_path:
    #              return PathFindingStatus.SUCCESSFUL
        
    #     if depth < 3 : return PathFindingStatus.FAILED_TIMEOUT
    #     return PathFindingStatus.FAILED_OBSTACLE # Default to fail

# --- Configuration ---
tracks = [f"tracks/track_0{i}.t" for i in range(1, 10)] # track_01.t to track_09.t

# Define parameter ranges for the grid search
# Alpha and Gamma: log-scaled
# np.logspace(start_exponent, end_exponent, num_points)
# e.g., -2 to 1 means 10^-2 (0.01) to 10^1 (10)
alpha_values = np.logspace(-2, 1, 4)  # e.g., [0.01, 0.1, 1.0, 10.0]
gamma_values = np.logspace(-2, 1, 4)  # e.g., [0.01, 0.1, 1.0, 10.0]

# Depth: integer values
depth_values = [3, 5, 7] # Example depths, adjust as needed

# Beta is fixed
beta_fixed = 0

param_grid_dict = {
    'alpha': alpha_values,
    'gamma': gamma_values,
    'depth': depth_values
}

grid = ParameterGrid(param_grid_dict)
num_combinations = len(list(grid)) # ParameterGrid is a generator, convert to list for len
print(f"Total parameter combinations to test: {num_combinations}")
print(f"Number of tracks: {len(tracks)}")
print(f"Total runs: {num_combinations * len(tracks)}")

# --- Run Optimization ---
results = []

for params in tqdm(grid, desc="Parameter Sets"):
    alpha = params['alpha']
    gamma = params['gamma']
    depth = params['depth']
    
    # Construct the parameters list for find_path: [alpha, beta, gamma]
    current_algo_params = [alpha, beta_fixed, gamma]
    
    successes_for_this_param_set = 0
    
    for track_path in tqdm(tracks, desc=f"Tracks (a={alpha:.2f},g={gamma:.2f},d={depth})", leave=False):
        try:
            outputpath = f"routes/{track_path.split('/')[-1].replace('.t', '')}_a{alpha:.2f}_g{gamma:.2f}_d{depth}.csv"
            
            status = find_path(
                track_path=track_path,
                visualize=False, # No visualization during optimization
                output=outputpath,     # No output files during optimization
                depth=depth,
                parameters=current_algo_params
            )
            
            is_successful = (status == PathFindingStatus.SUCCESSFUL)
            if is_successful:
                successes_for_this_param_set += 1
            
            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'depth': depth,
                'track': track_path,
                'status': status,
                'successful': is_successful
            })
            
        except Exception as e:
            print(f"Error running find_path for {track_path} with params {params}: {e}")
            results.append({
                'alpha': alpha,
                'gamma': gamma,
                'depth': depth,
                'track': track_path,
                'status': f"ERROR: {e}",
                'successful': False
            })
            
        #rename the output file in the visulizations folder
        # oritginal name is viusalizations/final_output.pdf
        
        # if the file exists:
        if os.path.exists("visualizations/final_output.pdf"):
            new_output_path = f"visualizations/{track_path.split('/')[-1].replace('.t', '')}_a{alpha:.2f}_g{gamma:.2f}_d{depth}.pdf"
            os.rename("visualizations/final_output.pdf", new_output_path)
            
# --- Analyze Results ---
if not results:
    print("No results were generated. Check for errors.")
else:
    df_results = pd.DataFrame(results)

    # Group by parameters and count total successes
    # The key for grouping will be a tuple of the parameters
    summary = df_results.groupby(['alpha', 'gamma', 'depth'])['successful'].sum().reset_index(name='total_successful_tracks')
    
    # Sort by the number of successful tracks in descending order
    summary_sorted = summary.sort_values(by='total_successful_tracks', ascending=False)

    print("\n--- Optimization Summary ---")
    print(f"Tested on {len(tracks)} tracks.")
    print(summary_sorted.to_string())

    if not summary_sorted.empty:
        best_params_row = summary_sorted.iloc[0]
        print("\n--- Best Performing Parameters ---")
        print(f"Alpha: {best_params_row['alpha']:.4f}")
        print(f"Gamma: {best_params_row['gamma']:.4f}")
        print(f"Depth: {best_params_row['depth']}")
        print(f"Successful on: {best_params_row['total_successful_tracks']}/{len(tracks)} tracks")
    else:
        print("\nNo successful runs found for any parameter combination.")

    # Optional: Save detailed results to CSV
    df_results.to_csv("optimization_results_detailed.csv", index=False)
    summary_sorted.to_csv("optimization_summary.csv", index=False)
    print("\nDetailed results saved to 'optimization_results_detailed.csv'")
    print("Summary saved to 'optimization_summary.csv'")