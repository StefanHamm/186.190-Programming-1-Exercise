import numpy as np
import os
import subprocess
import sys
import platform
import shlex # For safer command string joining/splitting
import shutil

# --- Functions running on the HOST ---

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
            return np.array([[]], dtype='U1').reshape(0, 0) # Returning empty for now

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
    #print(track)
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
                 print(" ".join(map(str, row))) # Fallback for other types/multi-char strings
    elif track.ndim == 1:
        print(" ".join(map(str, track))) # Handle 1D array
    else:
        print(track) # Fallback for other dimensions


# --- Function orchestrating Docker execution ---

def run_visualization_in_docker(
    trackFilePath: str,
    routeFilePath: str,
    outputPdfPath: str,
    docker_image: str = "tran-optim",
    intermediate_basename: str = "visualization_output" # Base name for .tex/.pdf inside container
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
    
    #check if the docker image exists , if not build it 
    try:
        subprocess.run(["docker", "image", "inspect", docker_image], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print(f"Docker image '{docker_image}' not found. Attempting to build it...")
        try:
            subprocess.run(["docker", "build", "-t", docker_image, "."], check=True)
            print(f"Successfully built Docker image '{docker_image}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image '{docker_image}': {e}")
            sys.exit(1)
    
    
    
    host_cwd = os.getcwd()
    container_workdir = "/app" # Standard workdir inside the container

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
        if ":" in mount_source: # e.g., C:/Users/...
            drive = mount_source[0].lower()
            mount_source = f"/{drive}{mount_source[2:]}" # /c/Users/...
    volume_map = f"{mount_source}:{container_workdir}"
    print(f"Volume mapping: {volume_map}")

    # --- Define Paths *Inside* the Container ---
    # These paths are relative to container_workdir (/app)
    # Use forward slashes for paths inside the container, even on Windows host
    container_viz_dir = "visualizations_temp" # Temp dir inside container
    container_tex_file = f"{container_viz_dir}/{intermediate_basename}.tex"
    container_pdf_file = f"{container_viz_dir}/{intermediate_basename}.pdf"
    container_log_file = f"{container_viz_dir}/{intermediate_basename}.log"
    container_perl_script = "src/visualise.pl" # Relative to /app
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
        "--rm",                      # Remove container after exit
        f"--volume={volume_map}",    # Mount host CWD to /app
        f"--workdir={container_workdir}", # Set working dir in container
        docker_image,                # The image name
        "sh", "-c",                  # Use shell to execute the command string
        cmd_inside_docker            # The commands to run inside
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
        print("--- Docker Container STDERR ---") # Print stderr even on success
        print(process.stderr)
        print("-------------------------------")
        print("Docker command finished successfully according to exit code.")

        # Final verification: Check if the output file exists on the HOST
        if os.path.exists(output_abs_path) and os.path.getsize(output_abs_path) > 0:
             print(f"\n[SUCCESS] Output PDF successfully created on host: {output_abs_path}")
        else:
             print(f"\n[WARNING] Docker command finished, but final PDF not found or empty on host: {output_abs_path}")
             print("Check container STDOUT/STDERR above for potential issues like file move errors inside the container.")
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
    #delete all files in the directory
    
    temp_viz_host_path = os.path.join(host_cwd, container_viz_dir)
    print(f"Attempting to remove: {temp_viz_host_path}")

    if os.path.exists(temp_viz_host_path) and os.path.isdir(temp_viz_host_path):
        # Optional safety check: ensure we are deleting the expected directory name
        # This helps prevent accidentally deleting something else if paths get mixed up.
        if os.path.basename(temp_viz_host_path) == container_viz_dir:
            try:
                shutil.rmtree(temp_viz_host_path) # Recursively remove the directory
                print(f"Successfully removed temporary directory: {temp_viz_host_path}")
            except OSError as e:
                print(f"[WARNING] Could not remove temporary directory {temp_viz_host_path}: {e}")
                print("         It might contain files that are in use, lack permissions, or other issues.")
                print("         You may need to remove it manually.")
        else:
            # This case should ideally not happen if code logic is correct
            print(f"[WARNING] Safety check failed: Path '{temp_viz_host_path}' does not end with expected name '{container_viz_dir}'. Cleanup aborted.")
    elif os.path.exists(temp_viz_host_path):
         # Path exists but is not a directory (unexpected)
         print(f"[WARNING] Path '{temp_viz_host_path}' exists but is not a directory. Cannot remove as directory.")
    else:
        # Directory not found, maybe it was already cleaned up inside container or never created due to earlier error
        print(f"Temporary directory '{temp_viz_host_path}' not found on host (already cleaned up or never created).")
# --- Main execution block (runs on HOST) ---

if __name__ == "__main__":
    # --- Setup Example Files/Dirs (if they don't exist) ---
    # Create dummy files for testing if they are missing
    os.makedirs("tracks", exist_ok=True)
    os.makedirs("routes", exist_ok=True)
    os.makedirs("src", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True) # Host dir for final output

    example_track_file = "tracks/track_02.t"
    example_route_file = "routes/exampleroute.csv"
    example_perl_script = "src/visualise.pl"
    example_output_pdf = "visualizations/final_output.pdf" # Different name for clarity

    # --- Actual script logic ---
    print("\n--- Loading and Displaying Track (on Host) ---")
    # Example using the dummy track file
    track = loadTrack(example_track_file)
    displayTrack(track)

    print("\n--- Running Visualization (inside Docker) ---")
    run_visualization_in_docker(
        trackFilePath=example_track_file,
        routeFilePath=example_route_file,
        outputPdfPath=example_output_pdf,
        docker_image="tran-optim" # Make sure this image exists and has perl + pdflatex
    )

    print("\n--- Script Finished ---")