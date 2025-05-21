import numpy as np
import os
import subprocess
import sys
import platform
import shlex  # For safer command string joining/splitting
import shutil
import pandas as pd
from src.state import CarState

from functools import lru_cache

# --- Functions running on the HOST ---
chsize =256000
class Track:
    def __init__(self, track: np.ndarray):
        """
        Initializes the Track object.
        :param track: A 2D numpy array representing the track layout,
                      where characters denote different elements ('F' for goal, 'S' for start).
        """
        if not isinstance(track, np.ndarray) or track.ndim != 2:
            raise ValueError("Input track must be a 2D numpy array.")
        self.track = track
        self.rows, self.cols = track.shape
        self.grass_coordinates = self.getGrassCoordinates()

        # Precompute boolean masks for each cell type for fast lookup
        self.is_obstacle: np.ndarray = (self.track == 'O')
        self.is_grass: np.ndarray = (self.track == 'G')
        self.is_track: np.ndarray = (self.track == 'T')
        self.is_goal: np.ndarray = (self.track == 'F')
        self.is_start: np.ndarray = (self.track == 'S')

    def getGoalCoordinates(self) -> list[tuple[int, int]]:
        """
        Returns a list of goal coordinates (row, column).
        Searches the track array for cells containing 'F'.
        """
        # np.where returns a tuple of arrays (one for each dimension)
        # containing the indices where the condition is true.
        goal_rows, goal_cols = np.where(self.track == 'F')

        # Zip the row and column arrays together to create coordinate tuples
        # and convert the result to a list.
        goal_coordinates = list(zip(goal_rows, goal_cols))
        return goal_coordinates

    def getGrassCoordinates(self) -> list[tuple[int, int]]:
        """
        Returns a list of grass coordinates
        Searches the track array for cells containing 'G'.
        """
        # np.where returns a tuple of arrays (one for each dimension)
        # containing the indices where the condition is true.
        grass_rows, grass_cols = np.where(self.track == 'G')

        # Zip the row and column arrays together to create coordinate tuples
        # and convert the result to a list.
        grass_coordinates = list(zip(grass_rows, grass_cols))
        return grass_coordinates

    def getStartCoordinates(self) -> tuple[int, int] | None:
        """
        Returns the start coordinate (row, column) or None if not found.
        Searches the track array for a cell containing 'S'.
        Assumes there is at most one start point.
        """
        start_rows, start_cols = np.where(self.track == 'S')

        if len(start_rows) > 0:
            # If one or more 'S' are found, return the coordinates of the first one.
            # We assume only one start based on typical usage.
            return (start_rows[0], start_cols[0])
        else:
            # No 'S' found in the track array.
            return None

    def getDistance(self, coord1: tuple[int, int], coord2: tuple[int, int]) -> float:
        """
        Returns the Euclidean distance between two coordinates (row, column).
        """
        # Ensure coordinates are numpy arrays for vectorized subtraction
        point1 = np.array(coord1)
        point2 = np.array(coord2)
        # Calculate Euclidean distance
        return np.linalg.norm(point1 - point2)
    @lru_cache(maxsize=chsize)
    def is_valid_coordinate(self, coord: tuple[int, int]) -> bool:
        """Checks if a coordinate (row, col) is within the track bounds."""
        row, col = coord
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_cell_type(self, coord: tuple[int, int]) -> str | None:
        """Returns the character type of the cell at the given coordinate."""
        if self.is_valid_coordinate(coord):
            row, col = coord
            return self.track[row, col]
        return None

    def get_neighbouring_cell_types(self, coord: tuple[int, int]) -> list[str]:
        """
        Returns a list of cell types for the 4 neighbouring cells (up, down, left, right).
        """
        row, col = coord
        neighbours = []
        # Check up
        if self.is_valid_coordinate((row - 1, col)):
            neighbours.append(self.track[row - 1, col])
        # Check down
        if self.is_valid_coordinate((row + 1, col)):
            neighbours.append(self.track[row + 1, col])
        # Check left
        if self.is_valid_coordinate((row, col - 1)):
            neighbours.append(self.track[row, col - 1])
        # Check right
        if self.is_valid_coordinate((row, col + 1)):
            neighbours.append(self.track[row, col + 1])
        return neighbours
    
        
    def getBrushedTrack(self, pathFile: str, brushSize: int = 4) -> 'Track':
        """
        Creates a new track by "painting" along a given path with a specified brush size.
        Assumes pathFile CSV format is: column_index, row_index_from_bottom.

        Args:
            pathFile (str): Path to CSV: col_idx,row_idx_from_bottom (e.g., "X,Y_cartesian").
            brushSize (int): Side length of the square brush. Must be 1 or greater.

        Returns:
            Track: A new Track object with the brushed track.
        """
        if brushSize < 1:
            raise ValueError("brushSize must be 1 or greater.")

        # 1. Load path from pathFile
        try:
            path_df = pd.read_csv(pathFile, header=None, comment='#')
            if path_df.shape[1] < 2:
                 raise ValueError("Path file must have at least two columns for column and row.")
            
            path_coords = []
            for _, row_data in path_df.iterrows():
                # CSV format is assumed to be: X (column), Y (row_from_bottom)
                # row_data.iloc[0] is X (column index)
                # row_data.iloc[1] is Y (row index, 0 at bottom, increasing upwards)
                
                col_from_csv = int(row_data.iloc[0])
                row_from_bottom_csv = int(row_data.iloc[1])

                # Convert row_from_bottom_csv to NumPy standard row index (0 at top)
                # self.rows is the total number of rows. Max index is self.rows - 1.
                # If row_from_bottom_csv is 0, numpy_row = self.rows - 1.
                # If row_from_bottom_csv is self.rows - 1, numpy_row = 0.
                numpy_standard_row = (self.rows - 1) - row_from_bottom_csv
                
                # Store coordinates in (numpy_standard_row, col_from_csv) format
                path_coords.append((numpy_standard_row, col_from_csv))

        except FileNotFoundError:
            raise FileNotFoundError(f"Path file {pathFile} not found.")
        except pd.errors.EmptyDataError:
            path_coords = []
        except Exception as e:
            raise ValueError(f"Error reading or parsing path file {pathFile}: {e}")

        brushed_track_arr = np.full(self.track.shape, fill_value='O', dtype=self.track.dtype)

        if not path_coords:
            return Track(brushed_track_arr)

        all_brushed_centers = set()

        def _apply_brush_at_point(r_center: int, c_center: int):
            # r_center, c_center are now standard NumPy coordinates
            if (r_center, c_center) in all_brushed_centers:
                return
            
            if not self.is_valid_coordinate((r_center, c_center)): # Checks bounds using NumPy coords
                return
            
            all_brushed_centers.add((r_center, c_center))

            start_offset = -((brushSize - 1) // 2)
            end_offset = (brushSize // 2) 
            
            for dr in range(start_offset, end_offset + 1):
                for dc in range(start_offset, end_offset + 1):
                    r_paint = r_center + dr
                    c_paint = c_center + dc

                    if self.is_valid_coordinate((r_paint, c_paint)):
                        brushed_track_arr[r_paint, c_paint] = self.track[r_paint, c_paint]
        
        if len(path_coords) == 1:
            r1, c1 = path_coords[0] # r1, c1 are already NumPy standard
            _apply_brush_at_point(r1, c1) # Corrected: (r1, c1) not (c1, r1)
            return Track(brushed_track_arr)

        for i in range(len(path_coords) - 1):
            # r1,c1 and r2,c2 are already NumPy standard coordinates
            r1, c1 = path_coords[i]
            r2, c2 = path_coords[i+1]

            if i == 0:
                _apply_brush_at_point(r1, c1)
            _apply_brush_at_point(r2, c2)

            dr_total = r2 - r1
            dc_total = c2 - c1

            if dr_total == 0 and dc_total == 0:
                continue

            num_steps = max(abs(dr_total), abs(dc_total))
            if num_steps == 0: # Should be caught by previous check, but for safety
                continue

            step_r = dr_total / num_steps
            step_c = dc_total / num_steps

            current_r, current_c = float(r1), float(c1)
            for _ in range(num_steps + 1): 
                r_line = int(round(current_r))
                c_line = int(round(current_c))
                
                _apply_brush_at_point(r_line, c_line) # r_line, c_line are NumPy standard
                
                current_r += step_r
                current_c += step_c
        
        return Track(brushed_track_arr)
    
    def saveToFile(self, path: str):
        """
        Save the track to a file.
        Each row of the track is written as a line in the file.
        """
        with open(path, 'w') as f:
            for row in self.track:
                f.write(''.join(row) + '\n')
        
        


def loadTrack(path: str) -> np.ndarray:
    """
    Load a track from a file where each character represents a cell.
    Reads the file line by line, handling potential whitespace.

    :param path: Path to the track file.
    :return: Track as a 2D numpy array of single characters (dtype='U1').
             Returns an empty 2D array (shape=(0,0)) if the file is empty,
             not found, or cannot be processed.
    """
    lines_data = []
    try:
        with open(path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace (including newline characters)
                stripped_line = line.strip()
                # Only process lines that are not empty after stripping
                if stripped_line:
                    # Convert the string line into a list of its individual characters
                    lines_data.append(list(stripped_line))

        if not lines_data:
            # Handle empty file or file with only whitespace
            print(f"Warning: Track file '{path}' is empty or contains only whitespace.")
            # Return an empty array with shape (0, 0)
            return np.array([[]], dtype='U1').reshape(0, 0)

        # Optional: Check for consistent line lengths (recommended for valid tracks)
        first_len = len(lines_data[0])
        if not all(len(row) == first_len for row in lines_data):
            # If lengths differ, NumPy will create an array with dtype=object,
            # which might cause issues later. It's often better to enforce consistency.
            print(f"Error: Track file '{path}' has inconsistent line lengths.")
            # Return an empty array or raise a ValueError
            # raise ValueError(f"Track file '{path}' has inconsistent line lengths.")
            return np.array([[]], dtype='U1').reshape(0, 0)  # Returning empty for now

        # Convert the list of character lists into a NumPy array
        # 'U1' dtype ensures each element is treated as a single Unicode character
        matrixTrack = np.array(lines_data, dtype='U1')
        return matrixTrack

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Track file not found at '{path}'") from e

    except Exception as e:
        raise Exception(f"Error loading track file '{path}': {e}") from e


def displayTrack(track: np.ndarray):
    """
    Display the track.
    :param track: Track as a numpy array.
    """
    # print(track)
    # Convert the track to a string representation
    if track.size == 0:
        print("<Empty Track>")
        return
    # Check if it's a 2D array of characters/strings
    if track.ndim == 2:
        for row in track:
            # Join elements in the row without spaces if they are single chars
            # Check the first element; assumes consistent rows
            if len(row) > 0 and isinstance(row[0], str) and len(row[0]) == 1:
                print("".join(row))
            else:
                print(" ".join(map(str, row)))  # Fallback for other types/multi-char strings
    elif track.ndim == 1:
        print(" ".join(map(str, track)))  # Handle 1D array
    else:
        print(track)  # Fallback for other dimensions


# --- Function orchestrating Docker execution ---

def run_visualization_in_docker(
        trackFilePath: str,
        routeFilePath: str,
        outputPdfPath: str,
        docker_image: str = "tran-optim",
        intermediate_basename: str = "visualization_output"  # Base name for .tex/.pdf inside container
):
    """
    Runs the Perl script and pdflatex inside a Docker container.

    Assumes this script is run from the project root directory on the host,
    and that 'src/visualise.pl', trackFilePath, and routeFilePath exist
    relative to this root. The outputPdfPath will also be created relative
    to this root on the host.

    Args:
        trackFilePath: Path to the track file (relative to host CWD).
        routeFilePath: Path to the route file (relative to host CWD).
        outputPdfPath: Desired output PDF path (relative to host CWD).
        docker_image: Name of the Docker image with Perl and pdflatex.
        intermediate_basename: Basename for intermediate files inside the container.
    """

    # check if the docker image exists , if not build it
    try:
        subprocess.run(["docker", "image", "inspect", docker_image], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print(f"Docker image '{docker_image}' not found. Attempting to build it...")
        try:
            subprocess.run(["docker", "build", "-t", docker_image, "."], check=True)
            print(f"Successfully built Docker image '{docker_image}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image '{docker_image}': {e}")
            sys.exit(1)

    host_cwd = os.getcwd()
    container_workdir = "/app"  # Standard workdir inside the container

    print(f"--- Preparing Docker Run ---")
    print(f"Host CWD: {host_cwd}")
    print(f"Container Workdir: {container_workdir}")
    print(f"Docker Image: {docker_image}")

    # --- Basic Host-side Checks ---
    perl_script_host_path = os.path.join("src", "visualise.pl")
    if not os.path.exists(perl_script_host_path):
        print(f"Error: Perl script 'src/visualise.pl' not found relative to {host_cwd}")
        sys.exit(1)

    track_abs_path = os.path.abspath(trackFilePath)
    if not os.path.exists(track_abs_path):
        print(f"Error: Host track file not found: {track_abs_path}")
        sys.exit(1)

    route_abs_path = os.path.abspath(routeFilePath)
    if not os.path.exists(route_abs_path):
        print(f"Error: Host route file not found: {route_abs_path}")
        sys.exit(1)

    # Ensure output directory exists on host *before* running docker
    # This is where the final PDF will appear via the volume mount.
    output_abs_path = os.path.abspath(outputPdfPath)
    output_dir_abs = os.path.dirname(output_abs_path)
    print(f"Ensuring host output directory exists: {output_dir_abs}")
    os.makedirs(output_dir_abs, exist_ok=True)
    print(f"Final output PDF expected on host at: {output_abs_path}")

    # --- Prepare Docker Volume Mount ---
    # Mount the host's current working directory to the container's workdir
    mount_source = host_cwd
    # Handle path format for Docker on Windows if necessary
    if platform.system() == "Windows":
        mount_source = mount_source.replace('\\', '/')
        if ":" in mount_source:  # e.g., C:/Users/...
            drive = mount_source[0].lower()
            mount_source = f"/{drive}{mount_source[2:]}"  # /c/Users/...
    volume_map = f"{mount_source}:{container_workdir}"
    print(f"Volume mapping: {volume_map}")

    # --- Define Paths *Inside* the Container ---
    # These paths are relative to container_workdir (/app)
    # Use forward slashes for paths inside the container, even on Windows host
    container_viz_dir = "visualizations_temp"  # Temp dir inside container
    container_tex_file = f"{container_viz_dir}/{intermediate_basename}.tex"
    container_pdf_file = f"{container_viz_dir}/{intermediate_basename}.pdf"
    container_log_file = f"{container_viz_dir}/{intermediate_basename}.log"
    container_perl_script = "src/visualise.pl"  # Relative to /app
    # Input/Output paths passed to commands should also be relative to /app
    container_track_file = os.path.relpath(track_abs_path, host_cwd).replace('\\', '/')
    container_route_file = os.path.relpath(route_abs_path, host_cwd).replace('\\', '/')
    container_output_pdf_target = os.path.relpath(output_abs_path, host_cwd).replace('\\', '/')

    # --- Command sequence to execute *inside* the container ---
    # Use 'sh -c' to run multiple commands, ensuring paths are quoted if they contain spaces
    # Use && to stop if a command fails
    # Use POSIX paths inside the command string
    cmd_inside_docker = f"""
        set -e
        echo "---> Running inside container: $(pwd) <---"
        echo "Perl script: {container_perl_script}"
        echo "Track file: {container_track_file}"
        echo "Route file: {container_route_file}"
        echo "Output Tex: {container_tex_file}"
        echo "Final PDF Target: {container_output_pdf_target}"

        # 1. Ensure visualization directory exists inside container
        mkdir -p {shlex.quote(container_viz_dir)}
        echo "---> Created dir {container_viz_dir}"

        # 2. Run Perl Script
        echo "---> Running Perl..."
        perl {shlex.quote(container_perl_script)} \
             {shlex.quote(container_track_file)} \
             {shlex.quote(container_route_file)} \
             {shlex.quote(container_tex_file)}
        echo "---> Perl finished."

        # 3. Check if Tex File was Created
        if [ ! -f {shlex.quote(container_tex_file)} ]; then
            echo "Error: Perl script did not create TEX file: {container_tex_file}"
            exit 1
        fi
        if [ ! -s {shlex.quote(container_tex_file)} ]; then
             echo "Error: TEX file is empty: {container_tex_file}"
             exit 1
        fi
        echo "---> TEX file found: {container_tex_file}"

        # 4. Run pdflatex
        echo "---> Running pdflatex..."
        pdflatex -interaction=nonstopmode \
                 -halt-on-error \
                 -file-line-error \
                 -output-directory={shlex.quote(container_viz_dir)} \
                 {shlex.quote(container_tex_file)}
        echo "---> pdflatex finished (exit code $?)." # $? might be 0 even if errors occurred, check log/pdf

        # 5. Check if PDF File was Created in the temp dir
        if [ ! -f {shlex.quote(container_pdf_file)} ]; then
            echo "Error: pdflatex did not create PDF file: {container_pdf_file}"
            echo "Checking log file: {container_log_file}"
            if [ -f {shlex.quote(container_log_file)} ]; then
                 echo "--- Log Start ---"
                 tail -n 50 {shlex.quote(container_log_file)}
                 echo "--- Log End ---"
            else
                 echo "Log file not found."
            fi
            exit 1
        fi
         if [ ! -s {shlex.quote(container_pdf_file)} ]; then
             echo "Error: PDF file is empty: {container_pdf_file}"
             exit 1
        fi
        echo "---> PDF file found: {container_pdf_file}"

        # 6. Move the generated PDF to the final desired location *relative* to /app
        # Ensure the target directory exists first
        mkdir -p $(dirname {shlex.quote(container_output_pdf_target)})
        mv {shlex.quote(container_pdf_file)} {shlex.quote(container_output_pdf_target)}
        echo "---> Moved PDF to final destination: {container_output_pdf_target}"

        # 7. Optional: Clean up intermediate files (tex, log, aux)
        # rm -f {shlex.quote(container_tex_file)} {shlex.quote(container_log_file)} {shlex.quote(container_viz_dir)}/*.aux

        echo "---> Container execution finished successfully."
    """

    # --- Construct the full docker run command ---
    docker_cmd = [
        "docker", "run",
        "--rm",  # Remove container after exit
        f"--volume={volume_map}",  # Mount host CWD to /app
        f"--workdir={container_workdir}",  # Set working dir in container
        docker_image,  # The image name
        "sh", "-c",  # Use shell to execute the command string
        cmd_inside_docker  # The commands to run inside
    ]

    print("-" * 30)
    print(f"Executing Docker Command:")
    # Print the command in a way that's easier to copy/paste if needed
    print(' '.join(shlex.quote(arg) for arg in docker_cmd[:-1]) + f" sh -c {shlex.quote(cmd_inside_docker)}")
    print("-" * 30)

    try:
        # Set a reasonable timeout for the entire docker process
        process = subprocess.run(docker_cmd, capture_output=True, text=True, check=True, timeout=180)
        print("--- Docker Container STDOUT ---")
        print(process.stdout)
        print("--- Docker Container STDERR ---")  # Print stderr even on success
        print(process.stderr)
        print("-------------------------------")
        print("Docker command finished successfully according to exit code.")

        # Final verification: Check if the output file exists on the HOST
        if os.path.exists(output_abs_path) and os.path.getsize(output_abs_path) > 0:
            print(f"\n[SUCCESS] Output PDF successfully created on host: {output_abs_path}")
        else:
            print(f"\n[WARNING] Docker command finished, but final PDF not found or empty on host: {output_abs_path}")
            print(
                "Check container STDOUT/STDERR above for potential issues like file move errors inside the container.")
            # Check if the temp visualization dir exists on host - it shouldn't if the container cleaned up
            temp_viz_host_path = os.path.join(host_cwd, container_viz_dir)
            if os.path.exists(temp_viz_host_path):
                print(f"Temporary directory '{container_viz_dir}' might still exist on host: {temp_viz_host_path}")


    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Docker command failed with return code {e.returncode}")
        print("--- Docker Container STDOUT ---")
        print(e.stdout)
        print("--- Docker Container STDERR ---")
        print(e.stderr)
        print("-------------------------------")
        sys.exit(1)
    except subprocess.TimeoutExpired as e:
        print(f"\n[ERROR] Docker command timed out after {e.timeout} seconds.")
        print("--- Docker Container STDOUT (if any) ---")
        print(e.stdout if e.stdout else "<No STDOUT captured>")
        print("--- Docker Container STDERR (if any) ---")
        print(e.stderr if e.stderr else "<No STDERR captured>")
        print("-------------------------------")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'docker' command not found. Is Docker installed, running, and in your system's PATH?")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred running docker: {e}")
        sys.exit(1)

    # delete the directory visulaization_temp
    # delete all files in the directory

    temp_viz_host_path = os.path.join(host_cwd, container_viz_dir)
    print(f"Attempting to remove: {temp_viz_host_path}")

    if os.path.exists(temp_viz_host_path) and os.path.isdir(temp_viz_host_path):
        # Optional safety check: ensure we are deleting the expected directory name
        # This helps prevent accidentally deleting something else if paths get mixed up.
        if os.path.basename(temp_viz_host_path) == container_viz_dir:
            try:
                shutil.rmtree(temp_viz_host_path)  # Recursively remove the directory
                print(f"Successfully removed temporary directory: {temp_viz_host_path}")
            except OSError as e:
                print(f"[WARNING] Could not remove temporary directory {temp_viz_host_path}: {e}")
                print("         It might contain files that are in use, lack permissions, or other issues.")
                print("         You may need to remove it manually.")
        else:
            # This case should ideally not happen if code logic is correct
            print(
                f"[WARNING] Safety check failed: Path '{temp_viz_host_path}' does not end with expected name '{container_viz_dir}'. Cleanup aborted.")
    elif os.path.exists(temp_viz_host_path):
        # Path exists but is not a directory (unexpected)
        print(f"[WARNING] Path '{temp_viz_host_path}' exists but is not a directory. Cannot remove as directory.")
    else:
        # Directory not found, maybe it was already cleaned up inside container or never created due to earlier error
        print(f"Temporary directory '{temp_viz_host_path}' not found on host (already cleaned up or never created).")


def bresenham_line(x0, y0, x1, y1):
    """
    Generates the grid cells that a line passes through from (x0, y0) to (x1, y1)
    using Bresenham's line algorithm.
    Returns a list of (row, col) tuples.
    """
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((y0, x0))  # row, col
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return cells


def normalize_map(data: np.ndarray) -> np.ndarray:
    normed = np.copy(data)
    finite_mask = np.isfinite(normed)
    if np.any(finite_mask):
        min_val = np.min(normed[finite_mask])
        max_val = np.max(normed[finite_mask])
        range_val = max_val - min_val
        if range_val > 0:
            normed[finite_mask] = (normed[finite_mask] - min_val) / range_val
        else:
            normed[finite_mask] = 1.0
    return normed


@lru_cache(maxsize=chsize)
def is_invalid_move(track: Track, from_state: CarState, to_state: CarState) -> bool:
    if not track.is_valid_coordinate((to_state.row, to_state.col)):
        return True

    min_row = min(from_state.row, to_state.row)
    max_row = max(from_state.row, to_state.row)
    min_col = min(from_state.col, to_state.col)
    max_col = max(from_state.col, to_state.col)

    for r in range(min_row - 1, max_row + 2):
        for c in range(min_col - 1, max_col + 2):
            if not track.is_valid_coordinate((r, c)):
                continue
            if track.is_obstacle[r, c]:
                if liang_barsky_intersect(c - 0.5,
                                          r - 0.5,
                                          c + 0.5,
                                          r + 0.5,
                                          from_state.col,
                                          from_state.row,
                                          to_state.col,
                                          to_state.row
                                          ):
                    return True
    r, c = from_state.position()
    if track.is_grass[r, c]:
        if abs(from_state.v_row) >= 2 and abs(to_state.v_row) - abs(from_state.v_row) >= 0:
            return True
        if abs(from_state.v_col) >= 2 and abs(to_state.v_col) - abs(from_state.v_col) >= 0:
            return True
        if abs(from_state.v_row) == 1 and abs(to_state.v_row) > abs(from_state.v_row):
            return True
        if abs(from_state.v_col) == 1 and abs(to_state.v_col) > abs(from_state.v_col):
            return True

    if from_state.position() == to_state.position():
        return True

    return False

@lru_cache(maxsize=chsize)
def liang_barsky_intersect(x_min, y_min, x_max, y_max, x1, y1, x2, y2):
    # based on https://www.geeksforgeeks.org/liang-barsky-algorithm/
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]
    t_enter = 0.0
    t_exit = 1.0

    for i in range(4):
        if p[i] == 0:  # Check if line is parallel to the clipping boundary
            if q[i] < 0:
                return False  # Line is outside and parallel, so completely discarded
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > t_enter:
                    t_enter = t
            else:
                if t < t_exit:
                    t_exit = t

    if t_enter > t_exit:
        return False  # Line is completely outside

    return True

# def is_invalid_move_new(track: Track, from_state: CarState, to_state: CarState) -> bool:
#     # Cache state attributes
#     from_r, from_c = from_state.row, from_state.col
#     to_r, to_c = to_state.row, to_state.col
#     from_vr, from_vc = from_state.v_row, from_state.v_col
#     to_vr, to_vc = to_state.v_row, to_state.v_col

#     if not track.is_valid_coordinate((to_r, to_c)):
#         return True

#     # If no actual movement in terms of position
#     if from_r == to_r and from_c == to_c:
#         # Original code has "from_state.position() == to_state.position()" at the end.
#         # If this implies invalid if *only* velocity changes, move it here.
#         # If it means "no move at all", it might be better handled by this.
#         # Let's stick to the original placement for now unless clarified.
#         pass


#     # Obstacle Collision Check using Bresenham's line
#     # The line for Liang-Barsky is between cell centers.
#     # (from_c, from_r) are (x1, y1) and (to_c, to_r) are (x2, y2)
#     # Cell (r,c) has its center at (c,r) in (x,y) coordinates for Liang-Barsky.
#     # Cell (r,c) boundaries for Liang-Barsky: x_min=c-0.5, y_min=r-0.5, x_max=c+0.5, y_max=r+0.5.
    
#     # Cells to check are those whose centers are on or near the line path
#     # If from_state.row/col are guaranteed integers (cell indices)
#     line_path_cells = bresenham_line_cells(from_r, from_c, to_r, to_c)

#     for r_cell, c_cell in line_path_cells:
#         # The bresenham algorithm itself might generate cells slightly out if the line starts/ends
#         # near a boundary and goes 'out'. However, is_valid_coordinate should catch this.
#         # Also, the first cell (from_state) collision is implicitly handled if it's an obstacle.
#         if not track.is_valid_coordinate((r_cell, c_cell)):
#             # This case might indicate the path goes out of bounds.
#             # Depending on game rules, this might be an immediate invalid move.
#             # For now, we only care if an *obstacle* is hit.
#             # If to_state is valid, but path goes out and back in, this is complex.
#             # The original loop `min_row-1` etc. would check obstacles outside.
#             # Bresenham checks cells *on the path*. If path goes out of bounds and hits an
#             # 'imaginary' obstacle there, this won't detect it.
#             # But track.is_obstacle should only be true for valid coordinates.
#             continue

#         if track.is_obstacle[r_cell, c_cell]:
#             if liang_barsky_intersect(c_cell - 0.5, # x_min
#                                       r_cell - 0.5, # y_min
#                                       c_cell + 0.5, # x_max
#                                       r_cell + 0.5, # y_max
#                                       from_c,       # x1 (line start)
#                                       from_r,       # y1 (line start)
#                                       to_c,         # x2 (line end)
#                                       to_r          # y2 (line end)
#                                       ):
#                 return True
    
#     # Grass rule checks
#     # Only check grass rules if starting on grass
#     # The original code uses from_state.position() which gives (row, col)
#     start_pos_r, start_pos_c = from_r, from_c # from_state.position() is from_r, from_c
#     if track.is_grass[start_pos_r, start_pos_c]:
#         abs_from_vr = abs(from_vr)
#         abs_to_vr = abs(to_vr)
#         abs_from_vc = abs(from_vc)
#         abs_to_vc = abs(to_vc)

#         # If current speed component is >= 2, cannot maintain or increase speed
#         if abs_from_vr >= 2 and abs_to_vr >= abs_from_vr:
#             return True
#         if abs_from_vc >= 2 and abs_to_vc >= abs_from_vc:
#             return True
        
#         # If current speed component is 1, cannot increase speed
#         if abs_from_vr == 1 and abs_to_vr > abs_from_vr:
#             return True
#         if abs_from_vc == 1 and abs_to_vc > abs_from_vc:
#             return True

#     # Final check: if the car hasn't moved position (e.g., only velocity changed, or tried to move to same spot)
#     # This check might be slightly different from checking (from_r, from_c) == (to_r, to_c)
#     # if CarState.position() involves more complex logic, but assuming it's direct row/col.
#     if from_r == to_r and from_c == to_c: # Effectively from_state.position() == to_state.position()
#         return True # Invalid if no change in position

#     return False

def get_line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """
    Get all integer grid cells that the line from (r0, c0) to (r1, c1) passes through.
    Uses Bresenham's line algorithm.
    (r0,c0) and (r1,c1) are cell indices (integers).
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    
    r, c = r0, c0
    
    # Determine step direction
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    
    # Decision parameter
    if dc > dr: # Slope < 1
        err = dc // 2
        for _ in range(dc):
            cells.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
    else: # Slope >= 1
        err = dr // 2
        for _ in range(dr):
            cells.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
            
    cells.append((r1, c1)) # Ensure endpoint is included
    return cells

# A slightly different Bresenham implementation that might be more common/robust:
def bresenham_line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 >= -dc: # Use >= for diagonal preference
            err -= dc
            r += sr
        if e2 <= dr:  # Use <= for diagonal preference
            err += dr
            c += sc
    return cells

# --- Main execution block (runs on HOST) ---

if __name__ == "__main__":
    # --- Setup Example Files/Dirs (if they don't exist) ---
    # Create dummy files for testing if they are missing
    # os.makedirs("tracks", exist_ok=True)
    # os.makedirs("routes", exist_ok=True)
    # os.makedirs("src", exist_ok=True)
    # os.makedirs("visualizations", exist_ok=True)  # Host dir for final output

    # example_track_file = "tracks/track_02.t"
    # example_route_file = "routes/exampleroute.csv"
    # example_perl_script = "src/visualise.pl"
    # example_output_pdf = "visualizations/final_output.pdf"  # Different name for clarity

    # # --- Actual script logic ---
    # print("\n--- Loading and Displaying Track (on Host) ---")
    # # Example using the dummy track file
    # track = loadTrack(example_track_file)
    # displayTrack(track)

    # print("\n--- Running Visualization (inside Docker) ---")
    # run_visualization_in_docker(
    #     trackFilePath=example_track_file,
    #     routeFilePath=example_route_file,
    #     outputPdfPath=example_output_pdf,
    #     docker_image="tran-optim"  # Make sure this image exists and has perl + pdflatex
    # )

    # print("\n--- Script Finished ---")

    # create a track object from track_03.t
    
    track = Track(loadTrack("tracks/track_03.t"))
    brushed_track = track.getBrushedTrack(brushSize=3, pathFile="routes/output.csv")
    
    # save the new track
    brushed_track.saveToFile("tracks/track_03_brushed.t")
    
    #now visualize the path on the new track
    
    run_visualization_in_docker(
        trackFilePath="tracks/track_03_brushed.t",
        routeFilePath="routes/output.csv",
        outputPdfPath="visualizations/track_03_brushed.pdf",
        docker_image="tran-optim"  # Make sure this image exists and has perl + pdflatex
    )
