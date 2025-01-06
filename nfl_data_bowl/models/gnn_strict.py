"""GNN Model with 2 layers and optional LSTM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv


class EnhancedGATBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim, heads, dropout, concat=False):
        super().__init__()
        self.gat = GATv2Conv(
            in_dim,
            hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.gat(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class RoutePredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        # x = F.gelu(x)
        # x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.conv3(x, edge_index)

        return x


class MultiRoutePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_route_classes,
        num_frames=1,
        num_heads=4,
        dropout=0.1,
        num_gnn_layers=3,
        num_positions=20,
        max_downs=4,
        max_quarters=5,
        num_teams=32,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_channels = hidden_dim
        self.num_route_classes = num_route_classes
        self.use_lstm = num_frames > 1

        # Position and feature embeddings
        position_emb_dim = 4
        self.position_embedding = nn.Embedding(num_positions, position_emb_dim)
        down_emb_dim = 2
        quarter_emb_dim = 2
        team_emb_dim = 8
        player_emb_dim = 16
        self.down_emb = nn.Embedding(max_downs, down_emb_dim)
        self.quarter_emb = nn.Embedding(max_quarters, quarter_emb_dim)
        self.team_emb = nn.Embedding(num_teams, team_emb_dim)
        self.player_emb = nn.Embedding(1000, player_emb_dim)

        # Calculate input dimension after embedding and concatenating features
        self.node_feature_dim = (
            16 + 4
        )  # 16 for position embedding + 1 for weight + 1 for is_offense + 1 for motion + eligibility

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.num_gnn_layers = num_gnn_layers
        i = 0
        for frame in range(num_frames):
            frame_layers = []
            for _ in range(num_gnn_layers):
                frame_layers.append(
                    EnhancedGATBlock(
                        hidden_dim,
                        hidden_dim,
                        edge_dim=3,
                        heads=2,
                        dropout=dropout,
                        concat=False,
                    )
                )
                # cannot use residual because shapes do not align
                frame_layers[0] = GATv2Conv(
                    5
                    + 4
                    + quarter_emb_dim
                    + down_emb_dim
                    + team_emb_dim
                    + player_emb_dim
                    + position_emb_dim
                    - 1,
                    hidden_dim,
                    edge_dim=3,
                    heads=2,
                    dropout=dropout,
                    concat=False,
                )
            self.gnn_layers.extend(frame_layers)

        self.gnn_layers[-1] = GATv2Conv(
            hidden_dim,
            self.num_route_classes,
            edge_dim=3,
            heads=2,
            dropout=dropout,
            concat=False,
        )

        # LSTM for temporal processing (only if num_frames > 1)
        if self.use_lstm:
            raise NotImplementedError()
        else:
            # For single frame, use a linear layer instead
            # For single frame, project from GAT output dim to final hidden dim
            self.frame_projection = None

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # Extract individual features from data.x
        position_idx = data.x[:, 0].long()  # Assuming position index is first column
        raw_features = data.x[:, 1:4]  # Rest of features (weight, is_offense, motion)

        # Create position embeddings
        position_embedded = self.position_embedding(
            position_idx
        )  # [num_nodes, hidden_dim]

        # Global features - indexed by batch
        down_embedded = self.down_emb(data.down - 1)  # [num_nodes, hidden_dim]
        quarter_embedded = self.quarter_emb(data.quarter - 1)  # [num_nodes, hidden_dim]
        team_embedded = self.team_emb(data.offense_team)  # [num_nodes, hidden_dim]
        player_embedded = self.player_emb(data.player_ids)

        game_clock = data.game_clock
        yardline = data.yardline
        yards_to_go = data.yards_to_go

        numeric_features = torch.cat(
            [
                game_clock.unsqueeze(1),
                yardline.unsqueeze(1),
                yards_to_go.unsqueeze(1),
                data.offense_score.unsqueeze(1),
                data.defense_score.unsqueeze(1),
            ],
            dim=1,
        )

        # Concatenate all features
        node_features = torch.cat(
            [
                position_embedded,  # [num_nodes, hidden_dim]
                raw_features,  # [num_nodes, num_raw_features]
                down_embedded,  # [num_nodes, hidden_dim]
                quarter_embedded,  # [num_nodes, hidden_dim]
                team_embedded,  # [num_nodes, hidden_dim]
                player_embedded,
                numeric_features,
            ],
            dim=1,
        )
        edge_index = data.edge_index
        edge_attr = data.edge_attr[:, 0:3]  # Keep edge features as is

        layer_idx = 0
        node_outputs = []
        for frame in range(self.num_frames):
            mask = edge_attr[:, 1] == frame
            frame_edge_index = edge_index[:, mask]
            frame_edge_attr = edge_attr[mask, :]
            for i in range(self.num_gnn_layers):
                node_features = self.gnn_layers[layer_idx](
                    node_features,
                    frame_edge_index,
                    frame_edge_attr,
                )
                layer_idx += 1
            node_outputs.append(node_features)

        if self.use_lstm:
            # Stack frame outputs for each node [num_nodes, num_frames, hidden_dim]
            node_temporal = torch.stack(node_outputs, dim=1)
            # Process temporal information for each node
            _, (hidden, _) = self.lstm(node_temporal)
            node_features = self.gelu(hidden[-1])  # [num_nodes, hidden_dim]
        else:
            # For single frame, use a projection that matches LSTM input dimensions
            # GAT output is [N, hidden_dim * num_heads], same as x2
            node_features = node_outputs[0]  # Shape: [N, hidden_dim * num_heads]

        eligible_predictions = node_features[data.eligible_mask, :]

        return {"route_predictions": eligible_predictions}
