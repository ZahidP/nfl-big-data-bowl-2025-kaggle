import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import entropy
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from tqdm import tqdm


def get_initial_center_position(frame_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Get the initial position of the center for normalization reference.
    Falls back to offensive line average if no center is found.

    Returns:
        Tuple of (x, y) coordinates for reference point
    """
    # Get the first frame
    first_frame = frame_data[frame_data["frameId"] == frame_data["frameId"].min()]

    # Find Center's position
    center = first_frame[first_frame["position"] == "C"]
    if len(center) == 1:
        return center["x"].iloc[0], center["y"].iloc[0]

    # Fallback to offensive line average if no Center is found
    offensive_line = first_frame[first_frame["position"].isin(["C", "G", "T", "OL"])]
    if len(offensive_line) > 0:
        return offensive_line["x"].mean(), offensive_line["y"].mean()

    # Final fallback to offensive players average
    offense = first_frame[first_frame["position_group"] == "Offense"]
    return offense["x"].mean(), offense["y"].mean()


def normalize_coordinates(
    frame_data: pd.DataFrame, play_direction: str, ref_x: float, ref_y: float
) -> pd.DataFrame:
    """
    Normalize player coordinates relative to the initial center position and play direction.
    """
    frame_data = frame_data.copy()

    # Normalize relative to reference point
    frame_data["x"] = frame_data["x"] - ref_x
    frame_data["y"] = frame_data["y"] - ref_y

    # Flip coordinates if play is going left
    if play_direction.lower() == "left":
        frame_data["x"] = -frame_data["x"]
        # frame_data["y"] = -frame_data["y"]

    return frame_data


def get_detailed_position_group(position: str) -> str:
    """
    Categorize players into specific position groups for more granular comparison.
    """
    position_groups = {
        "QB": "Quarterback",
        "C": "Offensive Line",
        "G": "Offensive Line",
        "T": "Offensive Line",
        "OL": "Offensive Line",
        "RB": "Skill Players",
        "FB": "Skill Players",
        "WR": "Skill Players",
        "TE": "Skill Players",
        "DE": "Defensive Line",
        "DT": "Defensive Line",
        "NT": "Defensive Line",
        "ILB": "Linebackers",
        "OLB": "Linebackers",
        "MLB": "Linebackers",
        "LB": "Linebackers",
        "CB": "Secondary",
        "S": "Secondary",
        "FS": "Secondary",
        "SS": "Secondary",
        "DB": "Secondary",
    }
    return position_groups.get(position, "Other")


def get_players_by_frame(
    play_df: pd.DataFrame, frame_start: int = None, frame_count: int = None
) -> dict:
    """
    Convert play DataFrame into frame-indexed dictionary of normalized player coordinates by position group.
    """
    # Add position group column
    play_df = play_df.copy()
    play_df["position_group"] = play_df["position"].map(get_detailed_position_group)

    # Get play direction
    play_direction = play_df["playDirection"].iloc[0]

    # Get initial center position for normalization
    ref_x, ref_y = get_initial_center_position(play_df)

    # Get frame range if specified
    if frame_start is not None:
        frame_ids = sorted(play_df["frameId"].unique())
        if frame_start < 0:  # Handle negative indexing
            frame_start = len(frame_ids) + frame_start
        frame_start = max(0, min(frame_start, len(frame_ids) - 1))

        if frame_count is not None:
            frame_ids = frame_ids[frame_start : frame_start + frame_count]
        else:
            frame_ids = frame_ids[frame_start:]

        play_df = play_df[play_df["frameId"].isin(frame_ids)]

    frames = {}
    # Group by frameId for efficiency
    for frame_id, frame_data in play_df.groupby("frameId"):
        # Normalize coordinates for this frame
        normalized_frame = normalize_coordinates(
            frame_data, play_direction, ref_x, ref_y
        )

        frames[frame_id] = {
            "Quarterback": [],
            "Offensive Line": [],
            "Skill Players": [],
            "Defensive Line": [],
            "Linebackers": [],
            "Secondary": [],
            "Other": [],
        }

        # Then group by position_group
        for group, group_data in normalized_frame.groupby("position_group"):
            frames[frame_id][group] = group_data[
                ["x", "y", "nflId", "club", "position"]
            ].to_dict("records")

    return frames


def analyze_receiver_positions(
    frame_data: dict, relative_distances: bool = True
) -> Dict[str, Dict]:
    """
    Analyze the positions and distances between receivers (WR/TE) in a frame.

    Args:
        frame_data: Dictionary containing player positions by position group
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position

    Returns:
        Dictionary containing:
        - wr_positions: Dict mapping nflId to their position (1-5)
        - distances: Dict mapping nflId to their distances to all receivers
        - nearest_distances: Dict mapping nflId to their closest receiver distance
        - center_distances: Dict mapping nflId to their signed distance from center
    """
    # Extract receivers from skill players group
    receivers = [
        player
        for player in frame_data["Skill Players"]
        if player["position"] in ("WR", "TE")
    ]

    if not receivers:
        return {
            "wr_positions": {},
            "distances": {},
            "nearest_distances": {},
            "center_distances": {},
        }

    # Convert to numpy array for vectorized calculations
    receiver_coords = np.array([[r["x"], r["y"]] for r in receivers])

    # Calculate pairwise distances between all receivers
    distances = np.sqrt(
        np.sum((receiver_coords[:, np.newaxis] - receiver_coords) ** 2, axis=2)
    )

    # Create distance mappings and positions
    distances_dict = {}
    wr_positions = {}
    nearest_distances = {}

    for i, receiver in enumerate(receivers):
        # Get distances to all receivers (will include self as 0)
        all_distances = distances[i]
        receiver_to_others = [(j, d) for j, d in enumerate(all_distances) if j != i]

        if relative_distances:
            # Sort other receivers by distance to current receiver while keeping original indices
            sorted_receivers = sorted(receiver_to_others, key=lambda x: x[1])

            # Map distances to receiver numbers (1-5 based on proximity)
            receiver_distances = {
                position + 1: distance
                for position, (_, distance) in enumerate(sorted_receivers)
            }

            # Store nearest distance (actual minimum distance)
            nearest_distances[receiver["nflId"]] = (
                sorted_receivers[0][1] if sorted_receivers else np.nan
            )

        else:
            # Sort receivers by Y position
            y_positions = sorted(range(len(receivers)), key=lambda x: receivers[x]["y"])
            position_map = {idx: pos + 1 for pos, idx in enumerate(y_positions)}

            # Map distances using Y-based positions
            receiver_distances = {position_map[j]: d for j, d in receiver_to_others}

            # Store Y-based position
            wr_positions[receiver["nflId"]] = position_map[i]

            # Store nearest distance (minimum of all distances)
            nearest_distances[receiver["nflId"]] = (
                min(d for _, d in receiver_to_others) if receiver_to_others else np.nan
            )

        distances_dict[receiver["nflId"]] = receiver_distances

    # Calculate signed distances from center (0, 0)
    center_distances = {
        receiver["nflId"]: receiver[
            "x"
        ]  # X-coordinate is already normalized relative to center
        for receiver in receivers
    }

    return {
        "wr_positions": wr_positions,
        "distances": distances_dict,
        "nearest_distances": nearest_distances,
        "center_distances": center_distances,
    }


def join_receiver_analysis_to_df(
    tracking_df: pd.DataFrame,
    game_id: int,
    play_id: int,
    relative_distances: bool = True,
    frame_data: dict = None,
) -> pd.DataFrame:
    """
    Join receiver analysis results to tracking DataFrame for a specific play.

    Args:
        tracking_df: Original tracking DataFrame
        game_id: Game ID to analyze
        play_id: Play ID to analyze
        relative_distances: If True, order receivers by distance to each receiver
                          If False, order receivers by absolute Y position
        frame_data: Optional pre-computed frame data

    Returns:
        DataFrame with receiver analysis columns added for WR/TE players
    """
    # Filter for specific game and play

    play_df = tracking_df[
        (tracking_df["gameId"] == game_id) & (tracking_df["playId"] == play_id)
    ].copy()

    if frame_data is None:
        frame_data = get_players_by_frame(play_df)

    # Initialize new columns
    if not relative_distances:
        play_df["wr_number"] = (
            np.nan
        )  # Which receiver they are (1-5 based on Y position)
    play_df["wr_center_distance"] = np.nan
    play_df["nearest_wr_distance"] = np.nan

    # Initialize distance columns
    for i in range(1, 6):
        play_df[f"dis_wr_{i}"] = np.nan

    # Process each frame
    for frame_id in play_df["frameId"].unique():
        if frame_id not in frame_data:
            continue

        analysis = analyze_receiver_positions(frame_data[frame_id], relative_distances)

        # Update values for receivers in this frame
        frame_mask = (play_df["frameId"] == frame_id) & (
            play_df["position"].isin(("WR", "TE", "RB"))
        )

        for nfl_id in play_df[frame_mask]["nflId"]:
            if nfl_id in analysis["distances"]:
                idx = play_df[
                    (play_df["frameId"] == frame_id) & (play_df["nflId"] == nfl_id)
                ].index

                # Set position number if using absolute Y positions
                if not relative_distances and nfl_id in analysis["wr_positions"]:
                    play_df.loc[idx, "wr_number"] = analysis["wr_positions"][nfl_id]

                # Set basic info
                play_df.loc[idx, "wr_center_distance"] = analysis["center_distances"][
                    nfl_id
                ]
                play_df.loc[idx, "nearest_wr_distance"] = analysis["nearest_distances"][
                    nfl_id
                ]

                # Set distances to other receivers
                for target_pos, distance in analysis["distances"][nfl_id].items():
                    play_df.loc[idx, f"dis_wr_{target_pos}"] = distance

    return play_df
