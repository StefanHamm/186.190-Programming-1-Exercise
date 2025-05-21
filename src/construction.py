# in this file we run the construction heuristcs of the tracks
from collections import deque

from matplotlib import pyplot as plt

from src.helper import loadTrack, run_visualization_in_docker, Track
from src.helper import normalize_map, is_invalid_move
from src.visualizer import PlotManager, animate_paths_pygame
import networkx as nx
import numpy as np
import argparse
import heapq
from src.state import CarState

plot_manager = PlotManager()  # Initialize the PlotManager


def precompute_goal_heuristic(track: Track):
    distance_map = np.full((track.rows, track.cols), np.inf)
    goals = track.getGoalCoordinates()
    visited = np.zeros((track.rows, track.cols), dtype=bool)

    heap = []

    for r, c in goals:
        distance_map[r, c] = 0
        heapq.heappush(heap, (0, r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        dist, r, c = heapq.heappop(heap)

        if visited[r, c]:
            continue
        visited[r, c] = True

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if not track.is_valid_coordinate((nr, nc)):
                continue
            if track.is_obstacle[nr, nc]:
                continue

            new_dist = dist + 1  # or use a different cost function if needed
            if new_dist < distance_map[nr, nc]:
                distance_map[nr, nc] = new_dist
                heapq.heappush(heap, (new_dist, nr, nc))

    return normalize_map(distance_map)


def get_weight(state: CarState, distance_map: np.ndarray, track: Track, alpha: float) -> float:
    row, col = state.row, state.col
    dist = distance_map[row, col]

    grass_penalty = 0.0
    if track.is_grass[state.position()]:
        grass_penalty = 5

    return alpha * dist + grass_penalty


def build_graph(track: Track, start_state: CarState, max_depth, distance_map, alpha: float, max_nodes_to_explore=1000):
    g = nx.DiGraph()
    queue = deque([(start_state, 0)])

    # Tracks states expanded (popped from queue) in THIS build_graph call
    visited_in_this_build = set()

    # Ensure start_state is in the graph, A* needs it.
    # It might become isolated if no valid moves are found.
    if not g.has_node(start_state):
        g.add_node(start_state)

    while queue:
        # Check computational budget BEFORE popping
        if len(visited_in_this_build) >= max_nodes_to_explore > 0:
            # print(f"  build_graph: Explored {len(visited_in_this_build)} states, limit {max_nodes_to_explore}. Halting expansion.")
            break

        current, depth = queue.popleft()

        # If already expanded this state in this build, skip.
        # (Should be rare if items added to queue carefully, but good safeguard)
        if current in visited_in_this_build:
            continue
        visited_in_this_build.add(current)

        if depth >= max_depth:
            continue

        for ax in [-1, 0, 1]:
            for ay in [-1, 0, 1]:

                new_vr = current.v_row + ax
                new_vc = current.v_col + ay
                new_r = current.row + new_vr
                new_c = current.col + new_vc
                new_state = CarState(new_r, new_c, new_vr, new_vc)

                # check if edge between current and new state is already in the graph
                # if g.has_edge(current, new_state):
                #     queue.append((new_state, depth + 1))
                #     continue

                if is_invalid_move(track, current, new_state):
                    continue

                weight = get_weight(new_state, distance_map, track, alpha)

                # Add edge (also adds nodes current and new_state to g if not already present)
                g.add_edge(current, new_state, weight=weight)

                if new_state not in visited_in_this_build:
                    # Add to queue only if it hasn't been expanded from yet in this build.
                    # (Note: new_state could already be IN the queue, added by another parent. BFS handles this.)
                    queue.append((new_state, depth + 1))
                # else:
                # Invalid transition (crash)
                # print(f"Invalid transition from {current} to {new_state}") # Can be verbose

    # print(f"  build_graph: Built graph with {len(g.nodes())} nodes, {len(g.edges())} edges. Explored {len(visited_in_this_build)} states.")
    # TODO: remove after debug
    # draw_graph_on_track(g, track)
    return g


def find_goal_node(graph, goals):
    for node in graph.nodes():
        if node.position() in goals:
            return node
    return None


def reached_goal(state: CarState, goals: list[tuple[int, int]]) -> bool:
    return state.position() in goals


def find_best_local_goal(track, graph, current_state, distance_map, alpha):
    best_node = None
    best_score = float('inf')
    best_goal_node = None
    best_goal_distance = float('inf')

    for node in graph.nodes:
        # Skip if node is the same as current to avoid self-loop
        if node == current_state:
            continue

        # Distance to goal
        d = distance_map[node.position()]
        if not np.isfinite(d):
            continue  # skip unreachable positions

        if d == 0.0:
            # It's a goal node → find the closest one to current_state
            dist_to_current = np.linalg.norm(np.array(node.position()) - np.array(current_state.position()))
            if dist_to_current < best_goal_distance:
                best_goal_distance = dist_to_current
                best_goal_node = node
            continue

        grass_penalty = 0.0
        if track.is_grass[node.position()]:
            grass_penalty = 5.0  # apply strong penalty to discourage grass

        score = alpha * d + grass_penalty

        if score < best_score:
            best_score = score
            best_node = node

    return best_goal_node if best_goal_node is not None else best_node


# add a enum for
# Aborting 
# No further graph could be built. Aborting.
# No reachable local goal found.
# No path found in this chunk.
# No valid path found.
# No start point 'S' found on the track!q
# No goal point(s) 'F' found on the track!
from enum import Enum


class PathFindingStatus(Enum):
    SUCCESS = "Success"
    ABORTING = "Aborting"
    NO_GRAPH_BUILT = "No further graph could be built. Aborting."
    NO_LOCAL_GOAL = "No reachable local goal found."
    NO_PATH_FOUND = "No path found in this chunk."
    NO_VALID_PATH = "No valid path found."


def solve_chunked_astar(track: Track, start_state: CarState, goals: list[tuple[int, int]], distance_map: np.ndarray,
                        max_depth_initial: int, visualize=False, alpha: float = None):
    current_state = start_state
    full_path = [current_state]
    if not alpha:
        alpha = 2.0

    timestep = 0

    current_operative_depth = max_depth_initial
    # This counter will now accumulate for ANY failure in a step (graph, local goal, A*)
    consecutive_step_failures = 0
    # MAX_CONSECUTIVE_GRAPH_FAILURES is now MAX_CONSECUTIVE_STEP_FAILURES
    # If this is e.g. 1, it will recover on the first failure. If >1, it retries current state.
    # Given build_graph is deterministic, >1 means repeated identical failures until recovery.
    # Consider setting this to 1 or 2 for faster recovery, or ensure retries offer variety.
    MAX_CONSECUTIVE_STEP_FAILURES = 3  # Tunable: How many times to fail consecutively before major recovery
    BACKTRACK_STEPS = 20
    DEPTH_INCREASE_STEP = 3
    MAX_TOTAL_TIMESTEPS = 450  # Increased timeout slightly
    pathOfPaths = []
    while not reached_goal(current_state, goals):
        timestep += 1
        if timestep > MAX_TOTAL_TIMESTEPS:  # Adjusted timeout
            print(f"Timeout reached ({MAX_TOTAL_TIMESTEPS} timesteps). Aborting.")
            return full_path, pathOfPaths, PathFindingStatus.ABORTING

        # print(f"T{timestep}: CS={current_state}, OpDepth={current_operative_depth}, StepFails={consecutive_step_failures}/{MAX_CONSECUTIVE_STEP_FAILURES}")

        graph = build_graph(track, current_state, current_operative_depth, distance_map, alpha,
                            max_nodes_to_explore=1000 + current_operative_depth * 200)  # Slightly more budget for deeper graphs

        if visualize and plot_manager:
            pathOfPaths.append(full_path.copy())
            plot_manager.create_graph_plot(
                graph, track, start_state=current_state,
                title=f"Graph T{timestep} (State: {current_state}, Depth: {current_operative_depth})"
            )

        step_planned_successfully = False
        # Check if graph is minimally valid (has the current_state, and potentially other nodes)
        # build_graph should ensure current_state is added. An empty graph means current_state wasn't even processable.
        if not graph or not graph.has_node(current_state) or len(graph.nodes()) == 0:
            print(f"  Graph build failed or unusable for state {current_state} with depth {current_operative_depth}.")
            # Failure will be handled below by incrementing consecutive_step_failures
        else:
            local_goal = find_best_local_goal(track, graph, current_state, distance_map, alpha)
            if not local_goal:
                print(f"  No reachable local goal found from {current_state} in the graph.")
            else:
                try:
                    partial_path = nx.astar_path(
                        graph,
                        current_state,
                        local_goal,
                        heuristic=lambda n, _: combined_heuristic(n, distance_map, alpha, track),
                        weight='weight'
                    )
                    if not partial_path or len(partial_path) < 2:
                        print(
                            f"  A* path planning issue: unexpected partial_path from {current_state} to {local_goal}.")
                    else:
                        # --- SUCCESSFUL PLANNING FOR THIS STEP ---
                        # print(f"  Successfully planned partial path to {local_goal}.")
                        consecutive_step_failures = 0  # Reset on full success of a step

                        # Optional: Logic to gradually decrease current_operative_depth
                        # if current_operative_depth > max_depth_initial and conditions_met:
                        #     current_operative_depth = max(max_depth_initial, current_operative_depth - 1)
                        #     print(f"  Gradually reducing depth to {current_operative_depth}")

                        if partial_path[-1].position() in goals:
                            # print("Goal reached directly by partial path!")
                            full_path.extend(partial_path[1:])
                            return full_path, pathOfPaths, PathFindingStatus.SUCCESS

                        full_path.append(partial_path[1])
                        current_state = partial_path[1]
                        step_planned_successfully = True

                except nx.NetworkXNoPath:
                    print(f"  No A* path in chunk from {current_state} to {local_goal}.")

        # --- Handle Step Outcome ---
        if step_planned_successfully:
            if reached_goal(current_state, goals):  # If the single step taken was to a goal cell
                return full_path, pathOfPaths, PathFindingStatus.SUCCESS
            continue  # Move to next timestep with the new current_state

        # --- If code reaches here, a failure occurred in planning this step ---
        consecutive_step_failures += 1
        print(f"  Step failed. Consecutive step failures: {consecutive_step_failures}/{MAX_CONSECUTIVE_STEP_FAILURES}.")

        if consecutive_step_failures >= MAX_CONSECUTIVE_STEP_FAILURES:
            print(f"  Max consecutive step failures reached. Attempting recovery (backtrack & depth increase).")

            state_before_recovery = current_state

            # Backtrack logic (adapted from your existing code)
            if len(full_path) > BACKTRACK_STEPS:
                print(f"  Backtracking {BACKTRACK_STEPS} steps from path of length {len(full_path)}.")
                full_path = full_path[:-BACKTRACK_STEPS]
                current_state = full_path[-1]
            elif len(full_path) > 1:
                print(f"  Path too short for full backtrack. Resetting to start of current segment: {full_path[0]}.")
                current_state = full_path[0]
                full_path = [current_state]
            else:
                current_state = state_before_recovery
                if not full_path or (full_path and full_path[-1] != current_state):
                    full_path = [current_state]
                print(f"  Path empty or too short to backtrack significantly. Retrying from {current_state}.")

            current_operative_depth += DEPTH_INCREASE_STEP
            print(f"  Increased lookahead depth to {current_operative_depth}.")
            consecutive_step_failures = 0  # Reset counter after taking recovery action
            # else:
            # Failed, but not enough times for major recovery.
            # The loop will retry from the *same current_state* with `consecutive_step_failures` incremented.
            # This means it will try to build_graph/plan again from the same spot.
            # If MAX_CONSECUTIVE_STEP_FAILURES is 1, this 'else' block is effectively skipped,
            # and recovery happens on the first failure. This might be more efficient if failures are deterministic.
            # If MAX_CONSECUTIVE_STEP_FAILURES > 1, it retries the same state, which is only useful
            # if failures are stochastic (e.g. related to max_nodes_to_explore budget being hit differently).
            # If failures are deterministic, set MAX_CONSECUTIVE_STEP_FAILURES to 1.
            pass  # Loop will continue, trying current_state again with incremented failure counter

    # Should be caught by `reached_goal` inside the loop or timeout
    return full_path, pathOfPaths, PathFindingStatus.SUCCESS  # Or appropriate status if loop exited unexpectedly


def combined_heuristic(
        state: CarState,
        distance_map: np.ndarray,
        alpha: float,
        track: Track
):
    row, col = state.row, state.col
    dist = distance_map[row, col]

    grass_penalty = 0.0

    if track.is_grass[state.position()]:
        grass_penalty = 5.0

    return alpha * dist + grass_penalty


def save_path_as_csv(path, output_path, track):
    with open(output_path, 'w', newline="\n") as f:
        for state in path:
            # adjust for different origins
            transformed_row = track.rows - 1 - state.row
            f.write(f"{state.col},{transformed_row}\n")


def find_path(track_path, visualize, output, depth, alpha: float = None):
    track = Track(loadTrack(track_path))

    start = track.getStartCoordinates()
    if start is None:
        print("No start point 'S' found on the track!")
        return

    goals = track.getGoalCoordinates()
    if not goals:
        print("No goal point(s) 'F' found on the track!")
        return

    start_state = CarState(start[0], start[1], 0, 0)

    print("Precomputing heuristic map...")
    distance_map = precompute_goal_heuristic(track)

    if visualize:
        plt.imshow(distance_map, cmap='viridis', origin='upper')
        plt.colorbar(label="Distance to Goal")
        plt.title("Precomputed Heuristic Map")
        plt.show()

    print("Running chunked A*...")
    path, paths, code = solve_chunked_astar(track, start_state, goals, distance_map, depth, visualize, alpha)

    if visualize:
        animate_paths_pygame(track=track, paths_list=paths)
        plot_manager.create_path_on_track_plot(track=track, path=path, title="Racetrack A* Result",
                                               show_acceleration=True)
        plot_manager.show_plots()  # Show all plots at once
    if path:
        print(f"Path found with {len(path)} steps.")
        save_path_as_csv(path, output, track)
        run_visualization_in_docker(
            trackFilePath=track_path,
            routeFilePath=output,
            outputPdfPath="visualizations/final_output.pdf"
        )
    else:
        print("No valid path found.")
    return code


def fast_path(path, alpha):
    track = Track(loadTrack(path))
    start = track.getStartCoordinates()
    start_state = CarState(start[0], start[1], 0, 0)
    goals = track.getGoalCoordinates()
    distance_map = precompute_goal_heuristic(track)
    depth = 3
    path, paths, code = solve_chunked_astar(track, start_state, goals, distance_map, depth, False, alpha)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run construction heuristic on a racetrack file.")
    parser.add_argument(
        "--track", "-t",
        type=str,
        default="tracks/track_02.t",
        help="Path to the track file. (default: tracks/track_02.t)"
    )

    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        default=False,
        help="Enable graphical visualization of the search process."
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="routes/output.csv",
        help="Path to save the output route file. (default: routes/output.csv)"
    )

    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=1,
        help="Maximum depth for graph building. (default: 1)"
    )

    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=2.0,
        help="Alpha parameter for the heuristic. (default: 2.0)"
    )

    args = parser.parse_args()

    find_path(
        track_path=args.track,
        visualize=args.visualize,
        output=args.output,
        depth=args.depth,
        alpha=args.alpha
    )
