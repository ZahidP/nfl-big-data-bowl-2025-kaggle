import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import random
import math
import torch.nn.functional as F
import sys
import os
from typing import Tuple, Dict, List
from torch_geometric.data import Data
from torch.utils.data import SubsetRandomSampler, DataLoader, Sampler
from collections import defaultdict
from multiprocessing import Pool, Manager
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool
from functools import partial
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from copy import deepcopy

from nfl_data_bowl.utils.graph_processing import (
    create_dynamic_frame_channel_graph,
    create_dynamic_frame_channel_graph_with_history,
)
from nfl_data_bowl.data_utils.data_common import (
    create_game_play_pairs,
    process_game_scores,
)


import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Tuple


class DataNormalizer:
    """
    Handles normalization of numerical features in the dataset.
    Supports both node features from DataFrame and dynamically generated edge features.
    """

    def __init__(self, method: str = "standardize"):
        """
        Initialize the normalizer.

        Args:
            method: Normalization method ('standardize' or 'minmax')
        """
        self.method = method
        self.node_stats = {}
        self.edge_stats = {}

        self.NODE_FEATURES = [
            "x",
            "y",
            "s",
            "a",
            "dis",  # Player movement features
            "yardsToGo",
            "gameClock",  # Game situation features
            "offense_score",
            "defense_score",  # Score features
        ]

        # Edge features will be populated during fit_edge_features
        self.edge_features = []

    def fit(self, df: pd.DataFrame) -> None:
        """
        Calculate normalization parameters from training data for node features.

        Args:
            df: Training dataframe
        """
        for feature in self.NODE_FEATURES:
            if feature in df.columns:
                if self.method == "standardize":
                    mean = df[feature].mean()
                    std = df[feature].std()
                    if std == 0:
                        std = 1  # Prevent division by zero
                    self.node_stats[feature] = {"mean": mean, "std": std}
                else:  # minmax
                    min_val = df[feature].min()
                    max_val = df[feature].max()
                    if min_val == max_val:
                        max_val = min_val + 1  # Prevent division by zero
                    self.node_stats[feature] = {"min": min_val, "max": max_val}

    def fit_edge_features(
        self, edge_features: torch.Tensor, feature_names: List[str]
    ) -> None:
        """
        Calculate normalization parameters for edge features.

        Args:
            edge_features: Tensor of edge features from multiple graphs [num_edges, num_features]
            feature_names: List of feature names corresponding to edge feature dimensions
        """
        self.edge_features = feature_names
        edge_features_np = edge_features.cpu().numpy()

        for idx, feature_name in enumerate(feature_names):
            feature_data = edge_features_np[:, idx]

            if self.method == "standardize":
                mean = float(np.mean(feature_data))
                std = float(np.std(feature_data))
                if std == 0:
                    std = 1
                self.edge_stats[feature_name] = {"mean": mean, "std": std}
            else:  # minmax
                min_val = float(np.min(feature_data))
                max_val = float(np.max(feature_data))
                if min_val == max_val:
                    max_val = min_val + 1
                self.edge_stats[feature_name] = {"min": min_val, "max": max_val}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to node features in the dataframe.

        Args:
            df: Input dataframe

        Returns:
            Normalized dataframe
        """
        df_normalized = df.copy()

        for feature in self.NODE_FEATURES:
            if feature in df.columns and feature in self.node_stats:
                if self.method == "standardize":
                    mean = self.node_stats[feature]["mean"]
                    std = self.node_stats[feature]["std"]
                    df_normalized[feature] = (df[feature] - mean) / std
                else:  # minmax
                    min_val = self.node_stats[feature]["min"]
                    max_val = self.node_stats[feature]["max"]
                    df_normalized[feature] = (df[feature] - min_val) / (
                        max_val - min_val
                    )

        return df_normalized

    def transform_edge_features(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to edge features.

        Args:
            edge_features: Edge feature tensor [num_edges, num_features]

        Returns:
            Normalized edge features tensor
        """
        if not self.edge_stats:
            return edge_features

        edge_features = edge_features.clone()
        device = edge_features.device

        for idx, feature_name in enumerate(self.edge_features):
            if feature_name in self.edge_stats:
                if self.method == "standardize":
                    mean = self.edge_stats[feature_name]["mean"]
                    std = self.edge_stats[feature_name]["std"]
                    edge_features[:, idx] = (edge_features[:, idx] - mean) / std
                else:  # minmax
                    min_val = self.edge_stats[feature_name]["min"]
                    max_val = self.edge_stats[feature_name]["max"]
                    edge_features[:, idx] = (edge_features[:, idx] - min_val) / (
                        max_val - min_val
                    )

        return edge_features.to(device)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the normalization for node features.

        Args:
            df: Normalized dataframe

        Returns:
            Original scale dataframe
        """
        df_original = df.copy()

        for feature in self.NODE_FEATURES:
            if feature in df.columns and feature in self.node_stats:
                if self.method == "standardize":
                    mean = self.node_stats[feature]["mean"]
                    std = self.node_stats[feature]["std"]
                    df_original[feature] = (df[feature] * std) + mean
                else:  # minmax
                    min_val = self.node_stats[feature]["min"]
                    max_val = self.node_stats[feature]["max"]
                    df_original[feature] = df[feature] * (max_val - min_val) + min_val

        return df_original

    def inverse_transform_edge_features(
        self, edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse the normalization for edge features.

        Args:
            edge_features: Normalized edge features tensor

        Returns:
            Original scale edge features tensor
        """
        if not self.edge_stats:
            return edge_features

        edge_features = edge_features.clone()
        device = edge_features.device

        for idx, feature_name in enumerate(self.edge_features):
            if feature_name in self.edge_stats:
                if self.method == "standardize":
                    mean = self.edge_stats[feature_name]["mean"]
                    std = self.edge_stats[feature_name]["std"]
                    edge_features[:, idx] = (edge_features[:, idx] * std) + mean
                else:  # minmax
                    min_val = self.edge_stats[feature_name]["min"]
                    max_val = self.edge_stats[feature_name]["max"]
                    edge_features[:, idx] = (
                        edge_features[:, idx] * (max_val - min_val) + min_val
                    )

        return edge_features.to(device)

    def get_node_stats(self) -> Dict:
        """Return the normalization parameters for node features."""
        return self.node_stats

    def get_edge_stats(self) -> Dict:
        """Return the normalization parameters for edge features."""
        return self.edge_stats


# Update DataMappings to include normalization
class DataMappings:
    def __init__(self):
        self.position_to_index = {}
        self.team_to_index = {}
        self.normalizer = DataNormalizer(method="standardize")

    def fit(self, df):
        """Create mappings from training data"""
        # Create position mapping
        unique_positions = sorted(df["position"].unique())
        self.position_to_index = {pos: idx for idx, pos in enumerate(unique_positions)}

        # Create team mapping
        unique_teams = sorted(df["possessionTeam"].unique())
        self.team_to_index = {team: idx for idx, team in enumerate(unique_teams)}

        # Fit normalizer for node features
        self.normalizer.fit(df)

    def fit_edge_features(self, edge_features: torch.Tensor, feature_names: List[str]):
        """Fit normalizer for edge features"""
        self.normalizer.fit_edge_features(edge_features, feature_names)

    def transform_data(self, df):
        """Apply normalization to node features"""
        return self.normalizer.transform(df)

    def transform_edge_features(self, edge_features: torch.Tensor):
        """Apply normalization to edge features"""
        return self.normalizer.transform_edge_features(edge_features)

    def inverse_transform_data(self, df):
        """Reverse normalization for node features"""
        return self.normalizer.inverse_transform(df)

    def inverse_transform_edge_features(self, edge_features: torch.Tensor):
        """Reverse normalization for edge features"""
        return self.normalizer.inverse_transform_edge_features(edge_features)

    def get_position_index(self, position):
        """Get index for position, return -1 if not found"""
        return self.position_to_index.get(position, -1)

    def get_team_index(self, team):
        """Get index for team, return -1 if not found"""
        return self.team_to_index.get(team, -1)

    @property
    def num_positions(self):
        return len(self.position_to_index)

    @property
    def num_teams(self):
        return len(self.team_to_index)


# class DataMappings:
#     def __init__(self):
#         self.position_to_index = {}
#         self.team_to_index = {}

#     def fit(self, df):
#         """Create mappings from training data"""
#         # Create position mapping
#         unique_positions = sorted(df["position"].unique())
#         self.position_to_index = {pos: idx for idx, pos in enumerate(unique_positions)}

#         # Create team mapping
#         unique_teams = sorted(df["possessionTeam"].unique())
#         self.team_to_index = {team: idx for idx, team in enumerate(unique_teams)}

#     def get_position_index(self, position):
#         """Get index for position, return -1 if not found"""
#         return self.position_to_index.get(position, -1)

#     def get_team_index(self, team):
#         """Get index for team, return -1 if not found"""
#         return self.team_to_index.get(team, -1)

#     @property
#     def num_positions(self):
#         return len(self.position_to_index)

#     @property
#     def num_teams(self):
#         return len(self.team_to_index)


def augment_play_chunk(args):
    """Helper function to augment a chunk of plays in parallel"""
    chunk, df, noise_std, do_not_augment = args
    augmented_data = []
    synthetic_pairs = []

    for game_id, play_id in chunk:
        play_data = df[(df["gameId"] == game_id) & (df["playId"] == play_id)].copy()

        if any(play_data.week.isin(do_not_augment)):
            continue

        # Generate synthetic data
        synthetic_data = play_data.copy()
        synthetic_data["y"] += np.random.normal(0, noise_std, len(synthetic_data))

        # Generate new unique IDs
        new_game_id = f"{game_id}_syn_0"
        new_play_id = f"{play_id}_syn_0"

        synthetic_data["gameId"] = new_game_id
        synthetic_data["playId"] = new_play_id

        augmented_data.append(synthetic_data)
        synthetic_pairs.append((new_game_id, new_play_id))

    return augmented_data, synthetic_pairs


class HistoricalMultiRoutePlayDataset(Dataset):
    def __init__(
        self,
        df,
        game_play_pairs,
        target_df,
        offense_positions,
        defense_positions,
        mappings,
        eligible_positions=["WR", "TE"],
        target_cols=None,
        n_frames=1,
        device="cuda",
        unique_routes=None,
        augment=False,
        do_not_augment_weeks=None,
        n_workers=None,
        teams_per_chunk=4,
        max_history_plays=5,
    ):
        super().__init__()

        # Store initial parameters
        self.df_original = df.copy()
        self.game_play_pairs = game_play_pairs
        self.n_frames = n_frames
        self.target_cols = target_cols
        self.device = device
        self.offense_positions = offense_positions
        self.defense_positions = defense_positions
        self.eligible_positions = eligible_positions
        self.unique_positions = list(df.position.unique())
        self.mappings = mappings
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.augment = augment
        self.do_not_augment_weeks = do_not_augment_weeks if do_not_augment_weeks else []
        self.teams_per_chunk = teams_per_chunk
        self.max_history_plays = max_history_plays
        self.data_by_unique_key = {}

        self.pair_to_idx = {
            (pair[0], pair[1]): idx for idx, pair in enumerate(self.game_play_pairs)
        }

        # Initialize route encoding
        self._initialize_route_encoding(df, unique_routes)

        # Initialize data processing
        self._initialize_dataset()

    def _initialize_route_encoding(self, df, unique_routes):
        """Initialize route encoding mapping"""
        if "routeRan" in df.columns:
            if not unique_routes:
                unique_routes = sorted(df["routeRan"].dropna().unique())
            else:
                unique_routes = sorted(unique_routes)
            self.route_to_idx = {route: idx for idx, route in enumerate(unique_routes)}
            self.idx_to_route = {idx: route for route, idx in self.route_to_idx.items()}
            self.num_route_classes = len(unique_routes)
        else:
            raise ValueError(
                "Dataset must contain 'routeRan' column for route prediction"
            )

    def _fit_normalizers(self):
        """Fit normalizers using discovered features"""
        print("Fitting normalizers...")

        # Calculate score and game clock statistics explicitly
        all_offense_scores = []
        all_defense_scores = []
        all_game_clocks = []

        for game_id, play_id in self.game_play_pairs:
            play_data = self.df[
                (self.df["gameId"] == game_id) & (self.df["playId"] == play_id)
            ]
            if len(play_data) > 0:
                play = play_data.iloc[0]
                # Get scores
                offense_score = play.offenseScore
                defense_score = play.defenseScore
                all_offense_scores.append(offense_score)
                all_defense_scores.append(defense_score)
                # Get game clock
                all_game_clocks.append(play.gameClock)

        # Add statistics explicitly to normalizer
        self.mappings.normalizer.node_stats["offense_score"] = {
            "mean": float(np.mean(all_offense_scores)),
            "std": float(np.std(all_offense_scores)) or 1.0,
        }
        self.mappings.normalizer.node_stats["defense_score"] = {
            "mean": float(np.mean(all_defense_scores)),
            "std": float(np.std(all_defense_scores)) or 1.0,
        }
        self.mappings.normalizer.node_stats["gameClock"] = {
            "mean": float(np.mean(all_game_clocks)),
            "std": float(np.std(all_game_clocks)) or 1.0,
        }

        # Fit node feature normalizer for other features
        self.mappings.fit(self.df)

    def _initialize_dataset(self):
        """Initialize the dataset with proper feature normalization sequence"""
        print("Initializing dataset and normalizers...")

        # Step 1: If augmentation is needed, do it first
        if self.augment:
            self.augment_play_data()
            self.df = self.df_original.copy()  # Work with augmented data
        else:
            self.df = self.df_original.copy()

        # Step 2: Initialize feature discovery with a small sample
        self._discover_features()

        # Step 3: Fit normalizers
        self._fit_normalizers()

        # Step 4: Process all plays with normalized features
        self._process_all_plays()

    def collect_edge_features(self):
        """
        Collect edge features from all plays to fit the normalizer.
        Should be called after initial graph creation but before final processing.
        """
        all_edge_features = []

        # Process a small batch of plays to get edge feature names
        sample_play = self.process_play_with_history(self.game_play_pairs[0], [])
        if sample_play and sample_play.frames:
            self.edge_feature_names = [
                f"edge_{i}" for i in range(sample_play.frames[0]["edge_attr"].size(1))
            ]

        # Collect edge features from all plays
        for game_id, play_id in tqdm(
            self.game_play_pairs, desc="Collecting edge features"
        ):
            play_data = self.df[
                (self.df["gameId"] == game_id) & (self.df["playId"] == play_id)
            ]
            graph = create_dynamic_frame_channel_graph(
                play_data,
                play_id,
                game_id,
                n_frames=self.n_frames,
                offense_positions=self.offense_positions,
                defense_positions=self.defense_positions,
                mappings=self.mappings,
            )

            for frame in graph["frames"]:
                all_edge_features.append(frame["edge_attr"])

        # Concatenate all edge features
        if all_edge_features:
            edge_features_tensor = torch.cat(all_edge_features, dim=0)
            # Fit the normalizer with collected edge features
            self.mappings.fit_edge_features(
                edge_features_tensor, self.edge_feature_names
            )

    def _discover_features(self):
        """Discover node and edge features from a small sample"""
        print("Discovering feature structure...")

        # Take a small sample of plays for feature discovery
        sample_size = min(10, len(self.game_play_pairs))
        sample_pairs = (
            self.game_play_pairs[:sample_size]
            + self.game_play_pairs[100 : 100 + sample_size]
        )

        # Create a sample graph to understand feature structure
        sample_play = self.game_play_pairs[0]
        play_data = self.df[
            (self.df["gameId"] == sample_play[0])
            & (self.df["playId"] == sample_play[1])
        ]

        sample_graph = create_dynamic_frame_channel_graph(
            play_data,
            sample_play[1],
            sample_play[0],
            n_frames=self.n_frames,
            offense_positions=self.offense_positions,
            defense_positions=self.defense_positions,
            mappings=self.mappings,
        )

        # Extract feature information
        if sample_graph["frames"]:
            self.edge_feature_names = [
                f"edge_{i}"
                for i in range(sample_graph["frames"][0]["edge_attr"].size(1))
            ]
            self.node_feature_names = [
                f"node_{i}"
                for i in range(sample_graph["frames"][0]["node_features"].size(1))
            ]

        # Store sample graphs for normalizer fitting
        self.sample_graphs = []
        for game_id, play_id in sample_pairs:
            play_data = self.df[
                (self.df["gameId"] == game_id) & (self.df["playId"] == play_id)
            ]
            graph = create_dynamic_frame_channel_graph(
                play_data,
                play_id,
                game_id,
                n_frames=self.n_frames,
                offense_positions=self.offense_positions,
                defense_positions=self.defense_positions,
                mappings=self.mappings,
            )
            self.sample_graphs.append(graph)

    # def _fit_normalizers(self):
    #     """Fit normalizers using discovered features"""
    #     print("Fitting normalizers...")

    #     # Fit node feature normalizer
    #     self.mappings.fit(self.df)

    #     # Collect edge features from sample graphs
    #     all_edge_features = []
    #     for graph in self.sample_graphs:
    #         for frame in graph["frames"]:
    #             all_edge_features.append(frame["edge_attr"])

    #     if all_edge_features:
    #         edge_features_tensor = torch.cat(all_edge_features, dim=0)
    #         self.mappings.fit_edge_features(edge_features_tensor, self.edge_feature_names)

    def group_plays_by_team(self):
        """Group plays by possession team and sort chronologically"""
        team_plays = defaultdict(list)

        # Create a DataFrame with just the necessary columns for sorting
        play_info = self.df[
            ["gameId", "playId", "possessionTeam", "time"]
        ].drop_duplicates()

        # Sort by time to ensure chronological order
        play_info_sorted = play_info.sort_values(["possessionTeam", "time"])

        # Group plays by team
        for _, row in play_info_sorted.iterrows():
            if (row["gameId"], row["playId"]) in self.pair_to_idx:
                team_plays[row["possessionTeam"]].append((row["gameId"], row["playId"]))

        return team_plays

    def chunk_teams(self, team_plays):
        """Split teams into chunks for parallel processing"""
        teams = list(team_plays.keys())
        n_chunks = max(1, len(teams) // self.teams_per_chunk)
        team_chunks = np.array_split(teams, n_chunks)

        chunks = []
        for team_chunk in team_chunks:
            chunk_dict = {team: team_plays[team] for team in team_chunk}
            chunks.append(chunk_dict)

        return chunks

    def _process_all_plays(self):
        """Process all plays with normalized features"""
        print("Processing all plays with normalized features...")

        # Group plays by team
        team_plays = self.group_plays_by_team()

        # Create chunks of teams
        chunks = self.chunk_teams(team_plays)

        # Process chunks with normalized features
        if self.n_workers > 1:
            with Manager() as manager:
                shared_play_bank = manager.dict()
                chunk_args = [(chunk, shared_play_bank) for chunk in chunks]

                with Pool(self.n_workers) as pool:
                    chunk_results = list(
                        tqdm(
                            pool.imap(self.process_team_chunk, chunk_args),
                            total=len(chunks),
                            desc="Processing team chunks",
                        )
                    )

                self.data_list = [
                    data for chunk_result in chunk_results for data in chunk_result
                ]
        else:
            chunk_results = [self.process_team_chunk((chunks[0], {}))]
            self.data_list = [
                data for chunk_result in chunk_results for data in chunk_result
            ]

        # Store processed data
        for data in self.data_list:
            key = f"{data['play_id']}_{data['game_id']}"
            self.data_by_unique_key[key] = data
            for frame in data.frames:
                frame["eligible_mask"] = frame["eligible_mask"].to(self.device)
                frame["route_targets"] = frame["route_targets"].to(self.device)

    def augment_play_data(self, noise_std=1.0):
        """Parallel implementation of play data augmentation"""
        print("Augmenting play data using multiprocessing")

        # Split the work into chunks for parallel processing
        chunk_size = max(1, len(self.game_play_pairs) // self.n_workers)
        chunks = [
            self.game_play_pairs[i : i + chunk_size]
            for i in range(0, len(self.game_play_pairs), chunk_size)
        ]

        # Prepare arguments for parallel processing
        augment_args = [
            (chunk, self.df_original, noise_std, self.do_not_augment_weeks)
            for chunk in chunks
        ]

        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(augment_play_chunk, augment_args),
                    total=len(chunks),
                    desc="Augmenting data chunks",
                )
            )

        # Combine results
        all_augmented_data = []
        all_synthetic_pairs = []
        for augmented_data, synthetic_pairs in results:
            all_augmented_data.extend(augmented_data)
            all_synthetic_pairs.extend(synthetic_pairs)

        # Update class attributes
        self.df = pd.concat([self.df_original] + all_augmented_data, ignore_index=True)
        self.game_play_pairs.extend(all_synthetic_pairs)

    def process_play_with_history(self, play_info, team_play_bank) -> Data:
        """Process a single play with normalized features"""
        game_id, play_id = play_info

        try:
            # Get and normalize play data
            play_data = self.df[
                (self.df["gameId"] == game_id) & (self.df["playId"] == play_id)
            ].copy()
            play_data = self.mappings.transform_data(play_data)

            # Get historical plays
            historical_plays = team_play_bank[-self.max_history_plays :]

            # Create graph with normalized features
            graph = create_dynamic_frame_channel_graph_with_history(
                play_data,
                historical_plays,
                play_id,
                game_id,
                n_frames=self.n_frames,
                offense_positions=self.offense_positions,
                defense_positions=self.defense_positions,
                mappings=self.mappings,
            )

            # Normalize game scores
            # if 'offense_score' in self.mappings.normalizer.node_stats:
            graph["offense_score"] = (
                graph["offense_score"]
                - self.mappings.normalizer.node_stats["offense_score"]["mean"]
            ) / self.mappings.normalizer.node_stats["offense_score"]["std"]
            graph["defense_score"] = (
                graph["defense_score"]
                - self.mappings.normalizer.node_stats["defense_score"]["mean"]
            ) / self.mappings.normalizer.node_stats["defense_score"]["std"]
            game_clock = torch.tensor(
                (
                    play_data.iloc[0].gameClock
                    - self.mappings.normalizer.node_stats["gameClock"]["mean"]
                )
                / self.mappings.normalizer.node_stats["gameClock"]["std"]
            )

            # Process frames
            processed_frames = []
            for frame_idx, frame in enumerate(graph["frames"]):
                # Normalize edge features
                normalized_edge_attr = self.mappings.transform_edge_features(
                    frame["edge_attr"]
                )

                frame_data = {
                    "node_features": frame["node_features"],
                    "edge_index": frame["edge_index"],
                    "edge_attr": normalized_edge_attr,
                }

                # Process eligibility and routes
                eligible_mask = []
                route_targets = []
                player_positions = []
                player_ids = []

                for i, (player_id, is_offense) in enumerate(
                    zip(frame["player_ids"], frame["node_features"][:, 2])
                ):
                    player_data = play_data[play_data["nflId"] == player_id].iloc[0]
                    is_eligible = (
                        player_data["position"] in self.eligible_positions
                        and "routeRan" in player_data
                    )
                    eligible_mask.append(is_eligible)
                    if is_eligible:
                        route_targets.append(
                            self.route_to_idx.get(player_data["routeRan"], -1)
                        )
                        player_positions.append(player_data["position"])
                        player_ids.append(player_id)
                    elif (
                        player_data["position"] in self.offense_positions
                        or player_data["position"] in self.defense_positions
                    ):
                        route_targets.append(-1)
                        player_positions.append(player_data["position"])
                        player_ids.append(player_id)

                frame_data.update(
                    {
                        "eligible_mask": torch.tensor(eligible_mask, dtype=torch.bool),
                        "route_targets": torch.tensor(route_targets, dtype=torch.long),
                        "player_positions": player_positions,
                        "player_ids": player_ids,
                    }
                )

                processed_frames.append(frame_data)

            # Create final data object
            data = Data(
                frames=processed_frames,
                historical_plays=historical_plays,
                game_id=game_id,
                play_id=play_id,
                quarter=torch.tensor(play_data.iloc[0].quarter),
                down=torch.tensor(play_data.iloc[0].down),
                game_clock=game_clock,
                time=torch.tensor(pd.Timestamp(play_data["time"].iloc[0]).timestamp()),
                yardsToGo=torch.tensor(play_data.iloc[0].yardsToGo),
                offense_score=torch.tensor(graph["offense_score"]),
                defense_score=torch.tensor(graph["defense_score"]),
                offense_team=graph["offense_team"],
                week=torch.tensor(play_data.iloc[0].week),
            )

            if any(frame["eligible_mask"].any() for frame in processed_frames):
                return data

        except Exception as e:
            print(f"Error processing play {play_id} for game {game_id}: {str(e)}")
            raise e

        return None

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    @property
    def num_features(self):
        return self.data_list[0].frames[0]["node_features"].size(1)

    @property
    def num_edge_features(self):
        return self.data_list[0].frames[0]["edge_attr"].size(1)

    def process_team_chunk(self, args):
        """Process a chunk of teams while maintaining sequential order"""
        chunk_dict, shared_play_bank = args
        processed_plays = []

        for team_id, plays in chunk_dict.items():
            team_play_bank = []

            # Process plays sequentially for this team
            for play_info in plays:
                # print(f'process_play_with_history: {i}')
                # sys.stdout.flush()
                # i += 1
                processed_play = self.process_play_with_history(
                    play_info, team_play_bank
                )
                if processed_play:
                    # TODO: this is in a weird place bc of data types
                    processed_plays.append(processed_play)
                    team_play_bank.append(processed_play)

                    # Update shared play bank
                    if team_id in shared_play_bank:
                        shared_play_bank[team_id].append(processed_play)
                        if len(shared_play_bank[team_id]) > self.max_history_plays:
                            shared_play_bank[team_id] = shared_play_bank[team_id][
                                -self.max_history_plays :
                            ]
                        else:
                            shared_play_bank[team_id] = [processed_play]

        return processed_plays
