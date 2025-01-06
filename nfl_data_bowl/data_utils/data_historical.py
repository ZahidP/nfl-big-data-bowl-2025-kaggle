import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
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


class DataMappings:
    def __init__(self):
        self.position_to_index = {}
        self.team_to_index = {}

    def fit(self, df):
        """Create mappings from training data"""
        # Create position mapping
        unique_positions = sorted(df["position"].unique())
        self.position_to_index = {pos: idx for idx, pos in enumerate(unique_positions)}

        # Create team mapping
        unique_teams = sorted(df["possessionTeam"].unique())
        self.team_to_index = {team: idx for idx, team in enumerate(unique_teams)}

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
        teams_per_chunk=4,  # New parameter for team chunking
        max_history_plays=5,  # New parameter for historical context
    ):
        super().__init__()
        self.df = df
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

        # Create route encoding mapping (same as before)
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

        self.game_play_pairs = game_play_pairs
        self.pair_to_idx = {
            (pair[0], pair[1]): idx for idx, pair in enumerate(self.game_play_pairs)
        }

        # Prepare target dataframe
        self.target_df = target_df
        if self.target_cols:
            self.target_df = self.target_df[["gameId", "playId"] + self.target_cols]

        # Precompute graphs
        self.precompute_graphs()

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

    def augment_play_data(self, noise_std=2.0):
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
            (chunk, self.df, noise_std, self.do_not_augment_weeks) for chunk in chunks
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
        self.df = pd.concat([self.df] + all_augmented_data, ignore_index=True)
        self.game_play_pairs.extend(all_synthetic_pairs)

    def process_play_with_history(self, play_info, team_play_bank) -> Data:
        """Process a single play with historical context"""
        game_id, play_id = play_info

        try:
            play_data = self.df[
                (self.df["gameId"] == game_id) & (self.df["playId"] == play_id)
            ]

            # Get historical plays for this team
            historical_plays = team_play_bank[-self.max_history_plays :]

            # Create base graph with historical context
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

            # Process frames (similar to original code)
            processed_frames = []
            for frame_idx, frame in enumerate(graph["frames"]):
                # print(type(frame))
                # print(frame_idx)
                # sys.stdout.flush()
                frame_data = {
                    "node_features": frame["node_features"],
                    "edge_index": frame["edge_index"],
                    "edge_attr": frame["edge_attr"],
                }

                # Create route target mask and values (same as original)
                eligible_mask = []
                route_targets = []
                player_positions = []
                player_ids = []

                for i, (player_id, is_offense) in enumerate(
                    zip(frame["player_ids"], frame["node_features"][:, 2])
                ):
                    player_data = play_data[play_data["nflId"] == player_id].iloc[0]
                    is_eligible = (
                        # is_offense == 1
                        player_data["position"] in self.eligible_positions
                        and "routeRan" in player_data
                    )
                    # TODO: ensure the offense and defense positions are tuples
                    eligible_mask.append(is_eligible)
                    if is_eligible:
                        route_targets.append(
                            self.route_to_idx.get(player_data["routeRan"], -1)
                        )
                        player_positions.append(player_data["position"])
                        player_ids.append(player_id)
                    elif (
                        player_data["position"] in self.offense_positions
                    ) or player_data["position"] in self.defense_positions:
                        route_targets.append(-1)
                        player_positions.append(player_data["position"])
                        player_ids.append(player_id)
                if torch.tensor(eligible_mask, dtype=torch.bool).sum() == 0:
                    raise Exception("wtf: %s", play_id)
                frame_data.update(
                    {
                        "eligible_mask": torch.tensor(eligible_mask, dtype=torch.bool),
                        "route_targets": torch.tensor(route_targets, dtype=torch.long),
                        "player_positions": player_positions,
                        "player_ids": player_ids,
                        # "historical_features": frame.get("historical_features", []),
                    }
                )

                processed_frames.append(frame_data)

            # Create final data object
            data = Data(
                frames=processed_frames,
                historical_plays=historical_plays,
                game_id=game_id,
                play_id=play_id,
                yardline=play_data.iloc[0].yardlineNumber,
                quarter=torch.tensor(play_data.iloc[0].quarter),
                down=torch.tensor(play_data.iloc[0].down),
                game_clock=torch.tensor(play_data.iloc[0].gameClock),
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
            print(frame)
            print(play_data)
            raise Exception(str(frame))

        return None

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

    def precompute_graphs(self):
        """Team-based parallel implementation of graph precomputation"""
        print("Precomputing graphs using team-based parallel processing")

        # First, augment the data if needed
        if self.augment:
            self.augment_play_data()

        # Group plays by team
        team_plays = self.group_plays_by_team()

        # Create chunks of teams
        chunks = self.chunk_teams(team_plays)

        # Create shared memory manager for play bank
        if self.n_workers > 1:
            with Manager() as manager:
                shared_play_bank = manager.dict()

                # Prepare chunk arguments
                chunk_args = [(chunk, shared_play_bank) for chunk in chunks]

                # Process chunks in parallel
                with Pool(self.n_workers) as pool:
                    chunk_results = list(
                        tqdm(
                            pool.imap(self.process_team_chunk, chunk_args),
                            total=len(chunks),
                            desc="Processing team chunks",
                        )
                    )

                # Combine results from all chunks
                self.data_list = [
                    data for chunk_result in chunk_results for data in chunk_result
                ]

                # Move data to specified device
                for data in self.data_list:
                    key = f"{data['play_id']}_{data['game_id']}"
                    self.data_by_unique_key[key] = data
                    for frame in data.frames:
                        frame["eligible_mask"] = frame["eligible_mask"].to(self.device)
                        frame["route_targets"] = frame["route_targets"].to(self.device)
                        if (
                            "historical_features" in frame
                            and frame["historical_features"] is not None
                        ):
                            frame["historical_features"] = frame[
                                "historical_features"
                            ].to(self.device)
        else:
            chunk_results = [self.process_team_chunk((chunks[0], {}))]
            self.data_list = [
                data for chunk_result in chunk_results for data in chunk_result
            ]
            for data in self.data_list:
                key = f"{data['play_id']}_{data['game_id']}"
                self.data_by_unique_key[key] = data
                for frame in data.frames:
                    frame["eligible_mask"] = frame["eligible_mask"].to(self.device)
                    frame["route_targets"] = frame["route_targets"].to(self.device)
                    if (
                        "historical_features" in frame
                        and frame["historical_features"] is not None
                    ):
                        frame["historical_features"] = frame["historical_features"].to(
                            self.device
                        )

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


def create_batch_data(plays_batch):
    """
    Helper function to create properly batched data for PyTorch Geometric.
    All features (except edge-related) are expanded to match the number of nodes.
    """
    batch_x = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_indices = []
    eligible_masks = []
    route_targets = []
    downs = []
    distances = []
    quarters = []
    offense_teams = []
    frame_indices = []
    times = []
    weeks = []
    plays_elapsed = []
    game_plays = []
    play_ids = []
    game_ids = []
    game_clock = []
    yards_to_go = []
    offense_score = []
    defense_score = []
    player_ids = []
    yardline = []

    node_offset = 0
    for batch_idx, play in enumerate(plays_batch):
        # Process historical plays
        for hist_idx, historical_play in enumerate(play["historical_plays"]):
            for frame_idx, frame in enumerate(historical_play["frames"]):
                num_nodes = frame["node_features"].size(0)

                # Add frame index to node features
                frame_idx = len(play["historical_plays"]) - hist_idx
                frame_idx_col = torch.full(
                    (num_nodes, 1),
                    frame_idx,
                    dtype=frame["node_features"].dtype,
                    device=frame["node_features"].device,
                )
                node_features_with_frame = torch.cat(
                    [frame["node_features"], frame_idx_col], dim=1
                )

                # Expand play-level features to match number of nodes
                yardline.extend([historical_play["yardline"].item()] * num_nodes)
                downs.extend([historical_play["down"].item()] * num_nodes)
                distances.extend([historical_play["yardsToGo"].item()] * num_nodes)
                quarters.extend([historical_play.get("quarter", 1)] * num_nodes)
                offense_teams.extend(
                    [historical_play.get("offense_team", 0)] * num_nodes
                )
                times.extend([historical_play["time"]] * num_nodes)
                weeks.extend([historical_play["week"]] * num_nodes)
                game_plays.extend([historical_play["play_id"]] * num_nodes)
                game_clock.extend([historical_play["game_clock"]] * num_nodes)
                yards_to_go.extend([historical_play["yardsToGo"]] * num_nodes)
                offense_score.extend([historical_play["offense_score"]] * num_nodes)
                defense_score.extend([historical_play["defense_score"]] * num_nodes)
                plays_elapsed.extend(
                    [len(play["historical_plays"]) - hist_idx] * num_nodes
                )

                # Node features and masks
                batch_x.append(node_features_with_frame)
                batch_indices.extend([batch_idx] * num_nodes)
                historical_eligible_mask = torch.zeros_like(frame["eligible_mask"])
                eligible_masks.append(historical_eligible_mask)

                # Edge features
                curr_edge_index = frame["edge_index"] + node_offset
                curr_edge_attr = frame["edge_attr"]

                # Add frame index to edge attributes
                frame_idx_col = torch.full(
                    (curr_edge_attr.size(0), 1),
                    frame_idx,
                    dtype=curr_edge_attr.dtype,
                    device=curr_edge_attr.device,
                )
                curr_edge_attr = torch.cat([curr_edge_attr, frame_idx_col], dim=1)

                batch_edge_index.append(curr_edge_index)
                batch_edge_attr.append(curr_edge_attr)
                frame_indices.extend([frame_idx] * curr_edge_index.size(1))

                route_targets.extend([-1] * num_nodes)  # Use -1 to indicate no target
                game_ids.extend([play["game_id"]] * num_nodes)
                play_ids.extend([play["play_id"]] * num_nodes)
                player_ids.extend(frame["player_ids"])
                node_offset += num_nodes

        # Process current play
        for frame_idx, frame in enumerate(play["frames"]):
            num_nodes = frame["node_features"].size(0)

            # Add frame index to node features
            frame_idx_col = torch.full(
                (num_nodes, 1),
                frame_idx,
                dtype=frame["node_features"].dtype,
                device=frame["node_features"].device,
            )
            node_features_with_frame = torch.cat(
                [frame["node_features"], frame_idx_col], dim=1
            )

            # Expand play-level features to match number of nodes
            downs.extend([play["down"].item()] * num_nodes)
            yardline.extend([play["yardline"].item()] * num_nodes)
            distances.extend([play["yardsToGo"].item()] * num_nodes)
            quarters.extend([play.get("quarter", 1)] * num_nodes)
            offense_teams.extend([play.get("offense_team", 0)] * num_nodes)
            times.extend([play["time"]] * num_nodes)
            weeks.extend([play["week"]] * num_nodes)
            game_plays.extend([play["play_id"]] * num_nodes)
            game_clock.extend([play["game_clock"]] * num_nodes)
            yards_to_go.extend([play["yardsToGo"]] * num_nodes)
            offense_score.extend([play["offense_score"]] * num_nodes)
            defense_score.extend([play["defense_score"]] * num_nodes)
            plays_elapsed.extend([0] * num_nodes)

            # Node features and masks
            batch_x.append(node_features_with_frame)
            batch_indices.extend([batch_idx] * num_nodes)
            eligible_masks.append(frame["eligible_mask"])

            # Edge features
            curr_edge_index = frame["edge_index"] + node_offset
            curr_edge_attr = frame["edge_attr"]

            # Add frame index to edge attributes
            frame_idx_col = torch.full(
                (curr_edge_attr.size(0), 1),
                frame_idx,
                dtype=curr_edge_attr.dtype,
                device=curr_edge_attr.device,
            )
            curr_edge_attr = torch.cat([curr_edge_attr, frame_idx_col], dim=1)

            batch_edge_index.append(curr_edge_index)
            batch_edge_attr.append(curr_edge_attr)
            frame_indices.extend([frame_idx] * curr_edge_index.size(1))

            route_targets.extend(frame["route_targets"])
            game_ids.extend([play["game_id"]] * num_nodes)
            play_ids.extend([play["play_id"]] * num_nodes)
            player_ids.extend(frame["player_ids"])
            node_offset += num_nodes

    if torch.cat(eligible_masks, dim=0).sum() == 0:
        raise Exception("No eligible receivers found in batch")

    # Create the batched data object with all features having same length as number of nodes
    return Data(
        x=torch.cat(batch_x, dim=0).to(torch.float),
        edge_index=torch.cat(batch_edge_index, dim=1),
        edge_attr=torch.cat(batch_edge_attr, dim=0),
        play_id=torch.tensor(play_ids),
        game_id=torch.tensor(game_ids),
        yardline=torch.tensor(yardline),
        batch=torch.tensor(batch_indices, dtype=torch.long),
        eligible_mask=torch.cat(eligible_masks, dim=0),
        time=torch.tensor(times, dtype=torch.float),
        route_targets=torch.tensor(route_targets, dtype=torch.long),
        down=torch.tensor(downs, dtype=torch.long),
        distance=torch.tensor(distances, dtype=torch.float),
        quarter=torch.tensor(quarters, dtype=torch.long),
        game_clock=torch.tensor(game_clock, dtype=torch.float),
        offense_team=torch.tensor(offense_teams, dtype=torch.long),
        offense_score=torch.tensor(offense_score, dtype=torch.float),
        defense_score=torch.tensor(defense_score, dtype=torch.float),
        week=torch.tensor(weeks, dtype=torch.long),
        yards_to_go=torch.tensor(yards_to_go, dtype=torch.float),
        player_ids=torch.tensor(player_ids, dtype=torch.long),
        plays_elapsed=torch.tensor(plays_elapsed, dtype=torch.long),
    )


def create_train_test_split(dataset, test_size=0.2, batch_size=32, random_seed=42):
    """
    Creates train and test dataloaders with stratified sampling for graph datasets.

    Parameters:
    - dataset: Graph-based dataset with multiple targets per row
    - test_size: Fraction of data to use for testing
    - batch_size: Batch size for dataloaders
    - random_seed: Random seed for reproducibility

    Returns:
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data
    - train_indices: Indices used for training
    - test_indices: Indices used for testing
    """
    # Set random seed
    np.random.seed(random_seed)

    # Shuffle the final index lists
    n_train = int(np.floor(len(dataset) * (1 - test_size)))
    train_indices = list(range(0, n_train))
    test_indices = list(range(n_train, len(dataset)))

    train_indices = np.random.permutation(train_indices)
    test_indices = np.random.permutation(test_indices)

    # Ensure no index exceeds dataset length
    train_indices = [idx for idx in train_indices if idx < len(dataset)]
    test_indices = [idx for idx in test_indices if idx < len(dataset)]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    print(f"Train set size: {len(train_indices)}")
    print(f"Test set size: {len(test_indices)}")

    return train_loader, test_loader, train_indices, test_indices


class TimeOrderedSampler(Sampler):
    """
    Sampler that maintains strict temporal ordering of indices
    while respecting batch size constraints and supporting multiple
    temporal ordering attributes.

    Parameters:
    - indices: List of indices to sample from
    - batch_size: Size of each batch
    - dataset: The dataset being sampled from
    - time_attribute: The attribute to use for temporal ordering ('time' or 'week')
    - drop_last: Whether to drop the last batch if it's smaller than batch_size
    """

    def __init__(
        self, indices, batch_size, dataset=None, time_attribute="time", drop_last=False
    ):
        self.indices = indices
        self.batch_size = batch_size
        self.dataset = dataset
        self.time_attribute = time_attribute
        self.drop_last = drop_last

        # If dataset is provided, sort indices by temporal attribute
        if dataset is not None:
            # Create (index, time_value) pairs
            time_pairs = []
            for idx in indices:
                batch = dataset[idx]
                time_value = getattr(batch, time_attribute)
                time_pairs.append((idx, time_value))

            # Sort indices by time value
            self.indices = [pair[0] for pair in sorted(time_pairs, key=lambda x: x[1])]

    def __iter__(self):
        # Calculate number of full batches
        n_batches = (
            len(self)
            if self.drop_last
            else (len(self.indices) + self.batch_size - 1) // self.batch_size
        )

        # Yield batches in temporal order
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.indices))
            if self.drop_last and end_idx - start_idx < self.batch_size:
                break
            yield self.indices[start_idx:end_idx]

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_time_stratified_split(
    dataset,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    batch_size=32,
    random_seed=42,
    intra_randomization=True,
    preserve_time_order=False,
    time_attribute="time",  # Can be 'time' or 'week'
):
    """
    Creates train, validation and test dataloaders with time-based stratification for graph datasets.

    Parameters:
    - dataset: Graph-based dataset with multiple targets per row and time information
    - train_size: Fraction of data to use for training (default: 0.7)
    - val_size: Fraction of data to use for validation (default: 0.15)
    - test_size: Fraction of data to use for testing (default: 0.15)
    - batch_size: Batch size for dataloaders
    - random_seed: Random seed for reproducibility
    - intra_randomization: Whether to randomize samples within each temporal group
    - preserve_time_order: If True, maintains strict temporal ordering in batches
    - time_attribute: Attribute to use for temporal ordering ('time' or 'week')

    Returns:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - test_loader: DataLoader for test data
    - split_indices: Dictionary containing train, val, and test indices
    - time_ranges: Dictionary containing time ranges for each split
    """
    assert (
        abs(train_size + val_size + test_size - 1.0) < 1e-5
    ), "Split proportions must sum to 1"
    np.random.seed(random_seed)

    # Create list of (index, time, targets) tuples
    indexed_data = []
    for i in range(len(dataset)):
        batch = dataset[i]
        time_value = getattr(batch, time_attribute)
        targets = batch.frames[-1]["route_targets"].cpu().numpy().flatten()
        # Only include indices with valid targets
        if np.any(targets != -1):
            indexed_data.append((i, time_value, targets))

    # Sort by time while keeping track of original indices
    sorted_data = sorted(indexed_data, key=lambda x: x[1])

    # Calculate split points
    train_split = int(len(sorted_data) * train_size)
    val_split = train_split + int(len(sorted_data) * val_size)

    # Split into train, validation and test
    train_data = sorted_data[:train_split]
    val_data = sorted_data[train_split:val_split]
    test_data = sorted_data[val_split:]

    # Extract indices and times
    train_indices = [item[0] for item in train_data]
    val_indices = [item[0] for item in val_data]
    test_indices = [item[0] for item in test_data]

    train_times = [item[1] for item in train_data]
    val_times = [item[1] for item in val_data]
    test_times = [item[1] for item in test_data]

    if preserve_time_order:
        # Use TimeOrderedSampler to maintain strict temporal ordering
        train_sampler = TimeOrderedSampler(train_indices, batch_size)
        val_sampler = TimeOrderedSampler(val_indices, batch_size)
        test_sampler = TimeOrderedSampler(test_indices, batch_size)
    else:
        # Optional intra-group randomization
        # Group indices by time
        train_time_groups = defaultdict(list)
        val_time_groups = defaultdict(list)
        test_time_groups = defaultdict(list)

        for idx, time, _ in train_data:
            train_time_groups[time].append(idx)
        for idx, time, _ in val_data:
            val_time_groups[time].append(idx)
        for idx, time, _ in test_data:
            test_time_groups[time].append(idx)

        # Shuffle within each time group and reconstruct indices
        train_indices = []
        val_indices = []
        test_indices = []

        for time in sorted(train_time_groups.keys()):
            group_indices = train_time_groups[time]
            if intra_randomization:
                np.random.shuffle(group_indices)
            train_indices.extend(group_indices)

        for time in sorted(val_time_groups.keys()):
            group_indices = val_time_groups[time]
            if intra_randomization:
                np.random.shuffle(group_indices)
            val_indices.extend(group_indices)

        for time in sorted(test_time_groups.keys()):
            group_indices = test_time_groups[time]
            if intra_randomization:
                np.random.shuffle(group_indices)
            test_indices.extend(group_indices)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size if not preserve_time_order else None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size if not preserve_time_order else None,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size if not preserve_time_order else None,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    # Print diagnostics
    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")
    print(f"Train {time_attribute} range: {min(train_times)} to {max(train_times)}")
    print(f"Validation {time_attribute} range: {min(val_times)} to {max(val_times)}")
    print(f"Test {time_attribute} range: {min(test_times)} to {max(test_times)}")

    # Additional validation checks
    train_max_time = max(train_times)
    val_min_time = min(val_times)
    val_max_time = max(val_times)
    test_min_time = min(test_times)

    assert (
        train_max_time <= val_min_time
    ), "Training data leaking into validation period!"
    assert val_max_time <= test_min_time, "Validation data leaking into test period!"

    # Verify ordering if requested
    if preserve_time_order:
        for loader_name, loader in [
            ("Train", train_loader),
            ("Val", val_loader),
            ("Test", test_loader),
        ]:
            prev_time = float("-inf")
            for batch in loader:
                curr_times = getattr(batch, time_attribute)
                assert torch.all(
                    curr_times >= prev_time
                ), f"{loader_name} loader violates temporal ordering!"
                prev_time = torch.max(curr_times)

    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}

    time_ranges = {
        "train": (min(train_times), max(train_times)),
        "val": (min(val_times), max(val_times)),
        "test": (min(test_times), max(test_times)),
    }

    return train_loader, val_loader, test_loader, split_indices, time_ranges


def create_week_stratified_split(
    dataset,
    train_weeks: list[int],
    val_weeks: list[int],
    test_weeks: list[int],
    batch_size=32,
    random_seed=42,
    preserve_time_order=False,
    val_random=True,  # Whether to randomly sample validation set from train period
):
    """
    Creates train, validation and test dataloaders with week-based stratification.

    Parameters:
    - dataset: Graph-based dataset with week information
    - train_weeks: List of weeks to use for training
    - val_weeks: List of weeks to use for validation
    - test_weeks: List of weeks to use for testing
    - batch_size: Batch size for dataloaders
    - random_seed: Random seed for reproducibility
    - preserve_time_order: If True, maintains strict temporal ordering in test set
    - val_random: If True, randomly samples validation set from training period
                 If False, uses specified val_weeks for time-stratified validation

    Returns:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - test_loader: DataLoader for test data
    - split_indices: Dictionary containing train, val, and test indices
    - week_ranges: Dictionary containing week ranges for each split
    """
    np.random.seed(random_seed)

    # Validate week lists
    all_weeks = set(train_weeks + val_weeks + test_weeks)

    # Create list of (index, week, time, targets) tuples
    indexed_data = []
    for i in range(len(dataset)):
        batch = dataset[i]
        week = batch.week
        time = batch.time
        targets = batch.frames[-1]["route_targets"].cpu().numpy().flatten()
        # Only include indices with valid targets
        if np.any(targets != -1):
            indexed_data.append((i, week, time, targets))

    # Split data by weeks
    train_data = [item for item in indexed_data if item[1] in train_weeks]
    test_data = [item for item in indexed_data if item[1] in test_weeks]

    if val_random:
        # Randomly sample validation set from training period
        n_val = int(len(train_data) * (len(val_weeks) / len(train_weeks)))
        np.random.shuffle(train_data)
        val_data = train_data[:n_val]
        train_data = train_data[n_val:]
    else:
        # Use time-stratified validation set
        val_data = [item for item in indexed_data if item[1] in val_weeks]

    # Sort test data by time to maintain temporal ordering
    test_data = sorted(test_data, key=lambda x: x[2])

    # Extract indices
    train_indices = [item[0] for item in train_data]
    val_indices = [item[0] for item in val_data]
    test_indices = [item[0] for item in test_data]

    # Create samplers
    if preserve_time_order:
        # Only use TimeOrderedSampler for test set
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = TimeOrderedSampler(
            test_indices, batch_size, dataset=dataset, time_attribute="time"
        )
    else:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size if not preserve_time_order else None,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=create_batch_data,
    )

    # Print diagnostics
    print(f"Train set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    print(f"Test set size: {len(test_indices)}")
    print(f"Train weeks: {sorted(train_weeks)}")
    if not val_random:
        print(f"Validation weeks: {sorted(val_weeks)}")
    else:
        print("Validation set randomly sampled from training period")
    print(f"Test weeks: {sorted(test_weeks)}")

    split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}

    week_ranges = {
        "train": (min(train_weeks), max(train_weeks)),
        "val": "random" if val_random else (min(val_weeks), max(val_weeks)),
        "test": (min(test_weeks), max(test_weeks)),
    }

    return train_loader, val_loader, test_loader, split_indices, week_ranges
