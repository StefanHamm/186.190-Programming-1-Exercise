# in this file we run the construction heuristcs of the tracks


from helper import loadTrack, displayTrack,run_visualization_in_docker,Track
import networkx as nx
import numpy as np
def generate_state_values(track_path):
    """
    Input: track
    Output: state values
    
    The output has the same size as the track but for each cell [x][y] we have the value of the state.
    This is used to find the best action between possible future actions.
    
    """
    
    track = Track(loadTrack(track_path))
    
    # create a np.array of the same size as the track
    # track class has some helper functions to get info about the track
    # e.g. start, goal coords and useful information
    
    
    state_value = np.zeros((track.cols, track.rows))
    
    print(state_value)






if __name__ == "__main__":
    
    example_track_file = "tracks/track_02.t"
    generate_state_values(example_track_file)