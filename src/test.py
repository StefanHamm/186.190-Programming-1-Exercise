import numpy as np
# from sklearn.model_selection import ParameterGrid # Not used in this version
from tqdm import tqdm # For progress bars, install with: pip install tqdm
# import pandas as pd # Not used in this version
import os
import time # Import the time module


from src.construction import find_path, PathFindingStatus,fast_path


def main():
    FAST = True # Set to True for faster runs, False for detailed runs
    # --- Configuration ---
    tracks_base_names = [f"track_0{i}" for i in range(2, 10)] # track_02 to track_09
    tracks_base_names.append("track_10")
    tracks = [f"tracks/{name}.t" for name in tracks_base_names]

    # Parameters for this run
    current_depth = 3
    current_parameters = [2, 0, 0] # alpha, beta, gamma

    # Clear out the routes folder and the visualizations folder
    if os.path.exists("routes"):
        print("Clearing routes folder...")
        for file_name in os.listdir("routes"):
            file_path = os.path.join("routes", file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs("routes", exist_ok=True)

    if os.path.exists("visualizations"):
        print("Clearing visualizations folder...")
        for file_name in os.listdir("visualizations"):
            file_path = os.path.join("visualizations", file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        os.makedirs("visualizations", exist_ok=True)


    print(f"\nProcessing tracks with depth={current_depth} and parameters={current_parameters}")
    results_summary = []

    for track_path in tqdm(tracks, desc="Processing Tracks"):
        track_name_no_ext = track_path.split('/')[-1].replace('.t', '')
        output_csv_path = f"routes/{track_name_no_ext}_d{current_depth}.csv"
        
        status = None # Initialize status
        duration = -1.0 # Initialize duration

        try:
            start_time = time.perf_counter() # Record start time
            if FAST:
                # Use a faster version of find_path for timing
                status_obj = fast_path(
                    path=track_path,
                    paramet=current_parameters
                )
            else:
                status_obj = find_path(
                    track_path=track_path,
                    visualize=False, # No visualization during timing runs
                    output=output_csv_path,
                    depth=current_depth,
                    parameters=current_parameters
                )
            # Ensure status is a string for consistent printing, especially if status_obj is an Enum
            status = status_obj.value if hasattr(status_obj, 'value') else str(status_obj)


            end_time = time.perf_counter() # Record end time
            duration = end_time - start_time # Calculate duration

            # print(f"Track: {track_path}, Status: {status}, Duration: {duration:.4f} seconds")
            
            # Rename the output PDF if it exists
            # Assuming find_path always creates "visualizations/final_output.pdf" if successful and visualize=True
            # (though visualize=False here, so this part might only be relevant if find_path still produces it)
            default_pdf_output = "visualizations/final_output.pdf"
            if os.path.exists(default_pdf_output):
                new_pdf_output_path = f"visualizations/{track_name_no_ext}_d{current_depth}.pdf"
                try:
                    os.rename(default_pdf_output, new_pdf_output_path)
                    # print(f"Renamed {default_pdf_output} to {new_pdf_output_path}")
                except Exception as e:
                    print(f"Error renaming PDF for {track_path}: {e}")
            
        except Exception as e:
            # This will catch errors from find_path OR from the timing/renaming logic itself
            print(f"An error occurred while processing {track_path}: {e}")
            status = "ERROR_IN_SCRIPT" # Indicate an error occurred during the processing loop
        
        finally:
            results_summary.append({
                "track": track_path,
                "status": status,
                "duration_seconds": duration
            })

    # Print summary of results
    print("\n--- Timing Summary ---")
    for result in results_summary:
        print(f"Track: {result['track']:<20} | Status: {str(result['status']):<30} | Duration: {result['duration_seconds']:.4f}s")

    print("\nProcessing complete.")
    
import cProfile
import os
if __name__ == "__main__":
    profile_file = "profile.out"
    cProfile.run('main()', profile_file)

    # Automatically launch snakeviz
    os.system(f"snakeviz {profile_file}")
        
        