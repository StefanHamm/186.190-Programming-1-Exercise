
import numpy as np
import os
import subprocess

def loadTrack(path: str) -> np.ndarray:
    """
    Load a track from a file.
    :param path: Path to the track file.
    :return: Track as a numpy array.
    """
    # the file consists of different symbols for different types of tracks
    # load the file and convert it into a n*m array.
    matrixTrack = np.loadtxt(path, dtype=str)
    return matrixTrack

def displayTrack(track: np.ndarray):
    """
    Display the track.
    :param track: Track as a numpy array.
    """
    # Convert the track to a string representation
    for row in track:
        print(row)
    
def runVisualization(trackFilePath: str, routeFilePath: str, outputPdfPath: str): # Changed output to PDF path
    """
    Run a perl script to generate tex, then run pdflatex.
    Generates a PDF file.
    """
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # Define paths and ensure directories exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    src_dir = "src"

    # Use os.path.join for cross-platform compatibility and absolute paths for robustness
    intermediateTexFile_rel = os.path.join(viz_dir, "output1.tex")
    intermediateTexFile_abs = os.path.abspath(intermediateTexFile_rel)
    perl_script_path = os.path.abspath(os.path.join(src_dir, "visualise.pl"))
    track_file_abs = os.path.abspath(trackFilePath)
    route_file_abs = os.path.abspath(routeFilePath)
    output_pdf_abs = os.path.abspath(outputPdfPath) # Define output PDF absolute path

    # --- Check Prerequisites ---
    if not os.path.exists(perl_script_path):
        print(f"Error: Perl script not found at {perl_script_path}")
        return
    if not os.path.exists(track_file_abs):
        print(f"Error: Track file not found at {track_file_abs}")
        return
    if not os.path.exists(route_file_abs):
        print(f"Error: Route file not found at {route_file_abs}")
        return

    # --- Run Perl Script ---
    cmd_perl = ["perl", perl_script_path, track_file_abs, route_file_abs, intermediateTexFile_abs]
    print(f"Running command: {' '.join(cmd_perl)}")
    try:
        # Use subprocess.run for simplicity and better error handling
        process_perl = subprocess.run(cmd_perl, capture_output=True, text=True, check=True, timeout=30)
        print(f"Perl STDOUT:\n{process_perl.stdout}")
        print(f"Perl STDERR:\n{process_perl.stderr}") # Print stderr too
    except subprocess.CalledProcessError as e:
        print(f"Perl script failed with return code {e.returncode}")
        print(f"Perl STDOUT:\n{e.stdout}")
        print(f"Perl STDERR:\n{e.stderr}")
        return # Stop if perl failed
    except subprocess.TimeoutExpired as e:
        print(f"Perl script timed out.")
        print(f"Perl STDOUT:\n{e.stdout}")
        print(f"Perl STDERR:\n{e.stderr}")
        return
    except FileNotFoundError:
        print(f"Error: 'perl' command not found. Is Perl installed and in your PATH?")
        return
    except Exception as e:
        print(f"An unexpected error occurred during Perl execution: {e}")
        return


    # --- Check if Tex File was Created ---
    if not os.path.exists(intermediateTexFile_abs):
        print(f"Error: Perl script did not create the TEX file: {intermediateTexFile_abs}")
        return
    if os.path.getsize(intermediateTexFile_abs) == 0:
        print(f"Error: TEX file is empty: {intermediateTexFile_abs}")
        # Optional: Delete empty file? os.remove(intermediateTexFile_abs)
        return
    print(f"TEX file created successfully: {intermediateTexFile_abs}")

    # --- Run pdflatex ---
    # Crucial flags:
    # -interaction=nonstopmode : Don't stop for errors, just report and continue if possible. PREVENTS THE '**' HANG.
    # -halt-on-error         : Stop processing after the first error is encountered (used with nonstopmode).
    # -file-line-error       : Show file and line number for errors.
    # -output-directory      : Explicitly set where output files (.pdf, .log, .aux) go. Helps with permissions and finding files.
    viz_dir_abs = os.path.abspath(viz_dir)
    cmd_latex = ["pdflatex",
                 "-interaction=nonstopmode",
                 "-halt-on-error",
                 "-file-line-error",
                 f"-output-directory={viz_dir_abs}",
                 intermediateTexFile_abs] # Pass the absolute path to the TeX file

    print(f"Running command: {' '.join(cmd_latex)}")
    log_file_path = os.path.join(viz_dir_abs, os.path.basename(intermediateTexFile_abs).replace('.tex', '.log')) # Predict log file path

    try:
        # Run pdflatex (might need to run twice for complex documents with references/TOC, but once is usually fine for TikZ)
        process_latex = subprocess.run(cmd_latex, capture_output=True, text=True, check=False, timeout=60) # Use check=False to analyze output even on failure

        print(f"pdflatex Return Code: {process_latex.returncode}")
        print(f"pdflatex STDOUT:\n{process_latex.stdout}")
        print(f"pdflatex STDERR:\n{process_latex.stderr}") # Usually empty, but check anyway

        # Check the log file for detailed errors, especially if return code is non-zero
        if process_latex.returncode != 0:
            print(f"\n--- pdflatex failed. Checking log file: {log_file_path} ---")
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f_log:
                        log_content = f_log.read()
                        # Look for common error indicators
                        error_lines = [line for line in log_content.splitlines() if line.startswith('! ')]
                        if error_lines:
                             print("Found errors in log:")
                             for err_line in error_lines:
                                 print(err_line)
                        else:
                             print("No lines starting with '! ' found in log, printing last 30 lines:")
                             print("\n".join(log_content.splitlines()[-30:]))

                except Exception as log_e:
                    print(f"Could not read log file {log_file_path}: {log_e}")
            else:
                print("pdflatex log file not found.")
            print("--- End pdflatex log check ---\n")
            print(f"! ==> Fatal error occurred during pdflatex execution, no output PDF file likely produced at expected location: {os.path.join(viz_dir_abs, os.path.basename(intermediateTexFile_abs).replace('.tex', '.pdf'))}")
            return # Stop if latex failed

        else:
            # Check if the expected PDF was actually created in the output directory
            generated_pdf_path = os.path.join(viz_dir_abs, os.path.basename(intermediateTexFile_abs).replace('.tex', '.pdf'))
            if os.path.exists(generated_pdf_path) and os.path.getsize(generated_pdf_path) > 0:
                 print(f"pdflatex completed successfully. PDF generated at: {generated_pdf_path}")
                 # Rename or move the generated PDF to the final desired output path if needed
                 try:
                     os.rename(generated_pdf_path, output_pdf_abs)
                     print(f"Moved PDF to final destination: {output_pdf_abs}")
                 except OSError as move_e:
                     print(f"Error moving PDF from {generated_pdf_path} to {output_pdf_abs}: {move_e}")

            else:
                 print(f"pdflatex returned success code, but the expected PDF file was not found or is empty: {generated_pdf_path}")
                 print("Check the pdflatex STDOUT above or the log file for warnings or subtle issues.")


    except subprocess.TimeoutExpired as e:
        print(f"pdflatex timed out after {e.timeout} seconds.")
        print(f"pdflatex STDOUT:\n{e.stdout}")
        print(f"pdflatex STDERR:\n{e.stderr}")
    except FileNotFoundError:
        print(f"Error: 'pdflatex' command not found. Is a LaTeX distribution (like MiKTeX or TeX Live) installed and in your PATH?")
    except Exception as e:
        print(f"An unexpected error occurred during pdflatex execution: {e}")

    
    
    
    
    
    

    
if __name__ == "__main__":
    
    for trackfile in os.listdir("tracks"):
        if trackfile.endswith(".t"):
            track = loadTrack(f"tracks/{trackfile}")
            displayTrack(track)
    
    # run the visualization
    runVisualization("tracks/track_02.t", "routes/exampleroute.csv", "visualizations/output1.png")
    
    