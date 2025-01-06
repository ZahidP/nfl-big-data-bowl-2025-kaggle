import pandas as pd
import torch
import math
import torch.nn.functional as F
import numpy as np


def get_game_scores(row):
    """
    Determine offensive and defensive scores based on possession and team information.

    Parameters:
    row: pandas Series containing game state information

    Returns:
    tuple: (offense_score, defense_score, possession_team)
    """
    possession_is_home = row["possessionTeam"] == row["homeTeamAbbr"]

    if possession_is_home:
        offense_score = row["preSnapHomeScore"]
        defense_score = row["preSnapVisitorScore"]
    else:
        offense_score = row["preSnapVisitorScore"]
        defense_score = row["preSnapHomeScore"]

    return offense_score, defense_score, row["possessionTeam"]


def process_game_scores(df):
    """
    Process entire dataframe to add offensive and defensive scores columns.

    Parameters:
    df: pandas DataFrame containing game state information

    Returns:
    pandas DataFrame: Original dataframe with two new columns added
    """
    # Create copy to avoid modifying original dataframe
    result_df = df.copy()

    # Apply get_game_scores to each row and create new columns
    result_df[["offenseScore", "defenseScore", "offenseTeam"]] = pd.DataFrame(
        result_df.apply(get_game_scores, axis=1).tolist(), index=result_df.index
    )

    return result_df


def calculate_distance_and_angle(node1, node2, play_direction):
    """Calculate distance and angle between two nodes."""
    dx = node2["x"] - node1["x"]
    dy = node2["y"] - node1["y"]
    dist = math.sqrt(dx**2 + dy**2)

    raw_angle = math.atan2(dy, dx)

    if play_direction == "right":
        angle = raw_angle
    else:
        angle = math.pi - raw_angle

    angle_deg = math.degrees(angle)
    angle_deg = (angle_deg + 360) % 360

    inverse_angle_deg = (angle_deg + 180) % 360

    return dist, angle_deg, inverse_angle_deg


def create_dynamic_frame_channel_graph(
    df,
    play_id,
    game_id,
    n_frames,
    offense_positions,
    defense_positions,
    mappings,
    emb_dim=4,
):
    """
    Create a dynamic graph representation of a play using consistent mappings.

    Parameters:
    df: DataFrame containing play data
    play_id: ID of the play to process
    game_id: ID of the game
    n_frames: Number of frames to process
    offense_positions: List of offensive positions to include
    defense_positions: List of defensive positions to include
    mappings: DataMappings object containing position and team mappings
    emb_dim: Embedding dimension

    Returns:
    dict: Complete graph representation of the play
    """
    df = df.copy()

    # Use provided mappings
    df["position_index"] = df["position"].map(mappings.position_to_index)
    df["team_index"] = df["possessionTeam"].map(mappings.team_to_index)

    play_data = df[(df["playId"] == play_id) & (df["gameId"] == game_id)].sort_values(
        "frameId"
    )
    frames = sorted(list(play_data["frameId"].unique()))[0:n_frames]

    # Initialize play-level information
    try:
        play_graph = {
            "play_id": play_id,
            "frames": [],
            "n_frames": n_frames,
            "actual_frames": len(frames),
            "quarter": play_data["quarter"].iloc[0],
            "down": play_data["down"].iloc[0],
            "yardsToGo": play_data["yardsToGo"].iloc[0],
            "gameClock": play_data["gameClock"].iloc[0],
            "time": pd.Timestamp(play_data["time"].iloc[0]).timestamp(),
            "absoluteYardlineNumber": play_data["absoluteYardlineNumber"].iloc[0],
            "offense_score": play_data["offenseScore"].iloc[0],
            "defense_score": play_data["defenseScore"].iloc[0],
            "offense_team": play_data["team_index"].iloc[0],
            "play_direction": play_data["playDirection"].iloc[0],
            "week": play_data["week"].iloc[0],
        }
    except Exception as e:
        print(f"Error processing play {play_id} in game {game_id}")
        raise e

    # Process each frame
    valid_frames = []
    for frame_idx, frame in enumerate(frames):
        frame_data = play_data[play_data["frameId"] == frame]
        frame_graph = process_single_frame(
            frame_data, offense_positions, defense_positions, frame_idx, mappings
        )
        if frame_graph is not None:  # Only append valid frames
            valid_frames.append(frame_graph)

    # If we have no valid frames, return None
    if not valid_frames:
        return None

    play_graph["frames"] = valid_frames

    return play_graph


def create_dynamic_frame_channel_graph_with_history(
    df,
    historical_data,
    play_id,
    game_id,
    n_frames,
    offense_positions,
    defense_positions,
    mappings,
    emb_dim=4,
):
    """
    Create a dynamic graph representation of a play using consistent mappings.

    Parameters:
    df: DataFrame containing play data
    play_id: ID of the play to process
    game_id: ID of the game
    n_frames: Number of frames to process
    offense_positions: List of offensive positions to include
    defense_positions: List of defensive positions to include
    mappings: DataMappings object containing position and team mappings
    emb_dim: Embedding dimension

    Returns:
    dict: Complete graph representation of the play
    """
    df = df.copy()

    # Use provided mappings
    df["position_index"] = df["position"].map(mappings.position_to_index)
    df["team_index"] = df["possessionTeam"].map(mappings.team_to_index)

    play_data = df[(df["playId"] == play_id) & (df["gameId"] == game_id)].sort_values(
        "frameId"
    )
    frames = sorted(list(play_data["frameId"].unique()))[0:n_frames]

    # Initialize play-level information
    try:
        play_graph = {
            "play_id": play_id,
            "frames": [],
            "historical_data": historical_data,
            "n_frames": n_frames,
            "actual_frames": len(frames),
            "quarter": play_data["quarter"].iloc[0],
            "down": play_data["down"].iloc[0],
            "yardsToGo": play_data["yardsToGo"].iloc[0],
            "gameClock": play_data["gameClock"].iloc[0],
            "time": pd.Timestamp(play_data["time"].iloc[0]).timestamp(),
            "absoluteYardlineNumber": play_data["absoluteYardlineNumber"].iloc[0],
            "offense_score": play_data["offenseScore"].iloc[0],
            "defense_score": play_data["defenseScore"].iloc[0],
            "offense_team": play_data["team_index"].iloc[0],
            "play_direction": play_data["playDirection"].iloc[0],
            "week": play_data["week"].iloc[0],
        }
    except Exception as e:
        print(f"Error processing play {play_id} in game {game_id}")
        raise e

    # Process each frame
    valid_frames = []

    for frame_idx, frame in enumerate(frames):
        frame_data = play_data[play_data["frameId"] == frame]
        frame_graph = process_single_frame(
            frame_data, offense_positions, defense_positions, frame_idx, mappings
        )
        if frame_graph is not None:  # Only append valid frames
            valid_frames.append(frame_graph)

    # If we have no valid frames, return None
    if not valid_frames:
        return None

    play_graph["frames"] = valid_frames

    return play_graph


def process_single_frame(
    frame_data: pd.DataFrame, offense_positions, defense_positions, frame_idx, mappings
):
    """
    Process a single frame of play data using consistent mappings.
    """
    if frame_data.empty:
        print(f"Empty frame data received for frame {frame_idx}")
        return None

    # # Debug print
    # print(f"\nProcessing frame {frame_idx}")
    # print(f"Original positions in frame: {frame_data['position'].unique()}")
    # print(f"Looking for offense positions: {offense_positions}")
    # print(f"Looking for defense positions: {defense_positions}")

    # Filter for offensive and defensive positions
    offense_mask = frame_data["position"].isin(offense_positions)
    defense_mask = frame_data["position"].isin(defense_positions)

    frame_data = (
        frame_data[offense_mask | defense_mask].copy().sort_values(by=["y", "x"])
    )

    frame_data = frame_data.sort_values(by=["y", "x"])

    if frame_data.empty:
        print(f"No players found after position filtering.")
        print(f"Number of offensive players: {offense_mask.sum()}")
        print(f"Number of defensive players: {defense_mask.sum()}")
        return None

    # Assign offense/defense labels
    frame_data["is_offense"] = (
        frame_data["position"].isin(offense_positions).astype(int)
    )

    # Get route information if available
    frame_data["route_encoded"] = None
    if "routeRan" in frame_data.columns:
        frame_data.loc[frame_data["routeRan"].notna(), "route_encoded"] = frame_data[
            "routeRan"
        ]

    # Create node features using position indices from mappings
    node_features = []
    for _, player in frame_data.iterrows():
        position_tensor = torch.tensor([player["position_index"]])
        weight_tensor = torch.tensor([player["weight"]])
        height_tensor = torch.tensor([player["height_inches"]])
        is_offense_tensor = torch.tensor([player["is_offense"]])
        motion_tensor = torch.tensor([np.abs(player["s"])])

        features = torch.cat(
            [
                position_tensor,
                weight_tensor,
                height_tensor,
                is_offense_tensor,
                motion_tensor,
            ]
        )
        node_features.append(features)

    # Create edges with permutation invariant angles
    edge_index = []
    edge_attr = []
    play_direction = frame_data["playDirection"].iloc[0]

    for i in range(len(frame_data)):
        for j in range(i + 1, len(frame_data)):
            node1 = frame_data.iloc[i]
            node2 = frame_data.iloc[j]

            # Calculate basic metrics
            dist, angle_deg, norm_vector = calculate_distance_and_angle(
                node1, node2, play_direction
            )

            if dist <= 100:  # Distance threshold
                # Add both directions with consistent angle representation
                edge_index.extend([[i, j], [j, i]])

                # For j->i edge, adjust angle by 180 degrees
                reverse_angle = (angle_deg + 180) % 360

                edge_attr.extend(
                    [[dist, frame_idx, angle_deg], [dist, frame_idx, reverse_angle]]
                )
            # if dist <= 100:  # Distance threshold
            #     # Add both directions with consistent angle representation
            #     edge_index.extend([[i, j]])

            #     # For j->i edge, adjust angle by 180 degrees
            #     reverse_angle = (angle_deg + 180) % 360

            #     edge_attr.extend([[dist, frame_idx, 1]])

    # Convert to tensors ensuring proper shape
    edge_index_tensor = (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr_tensor = (
        torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, 3))
    )

    xgb_cols = [c for c in frame_data.columns if "xgb" in c]

    frame_graph = {
        "node_features": (
            torch.stack(node_features) if node_features else torch.zeros((0, 4))
        ),
        "edge_index": edge_index_tensor,
        "edge_attr": edge_attr_tensor,
        "xgb_preds": torch.tensor(frame_data[xgb_cols].to_numpy()),
        "routes": frame_data["route_encoded"].tolist(),
        "player_ids": frame_data["nflId"].tolist(),
    }

    return frame_graph


def verify_permutation_invariance(
    frame_data, offense_positions, defense_positions, frame_idx, mappings
):
    """
    Verify that the angle calculations are permutation invariant.

    Returns:
        bool: True if permutation invariant, False otherwise
    """
    # Create original graph
    original_graph = process_single_frame(
        frame_data, offense_positions, defense_positions, frame_idx, mappings
    )

    # Create shuffled graph
    shuffled_data = frame_data.sample(frac=1).reset_index(drop=True)
    shuffled_graph = process_single_frame(
        shuffled_data, offense_positions, defense_positions, frame_idx, mappings
    )

    # Compare edge attributes after sorting
    orig_edges = set(tuple(e) for e in original_graph["edge_attr"].numpy())
    shuf_edges = set(tuple(e) for e in shuffled_graph["edge_attr"].numpy())

    return orig_edges == shuf_edges
