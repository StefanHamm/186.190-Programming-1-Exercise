
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
    
def runVisualization(trackFilePath: str, routeFilePath: str, outputFilePath: str):
    """
    Run a pearl script with the arguments track and route file paths.
    call /src/visualise.pl <trackFile> <tripFile> <outputFile>
    """
    # check perl -v 
    
    cmd = ["perl", "src/visualise.pl", trackFilePath, routeFilePath, outputFilePath]
    print(f"Running command: {' '.join(cmd)}")
    # run the command
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # check if the command was successful
    print(stdout)
    if process.returncode != 0:
        print(f"Error: {stderr.decode()}")
    else:
        print(f"Output: {stdout.decode()}")
    # run the visualization
    # check if the file was created
    
    
    
    
    

    
if __name__ == "__main__":
    
    for trackfile in os.listdir("tracks"):
        if trackfile.endswith(".t"):
            track = loadTrack(f"tracks/{trackfile}")
            displayTrack(track)
    
    # run the visualization
    runVisualization("tracks/track_02.t", "routes/exampleroute.csv", "visualization/output1.png")
    
    