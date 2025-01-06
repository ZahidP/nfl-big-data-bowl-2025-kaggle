import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch


# Add residual connections and layer normalization
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


class PlayGNN(nn.Module):
    def __init__(
        self,
        num_positions,
        hidden_dim=128,
        num_gnn_layers=3,
        num_route_classes=10,
        dropout=0.1,
        max_downs=4,
        max_quarters=4,
        num_teams=32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Global feature embeddings
        position_emb_dim = 4
        down_emb_dim = 2
        quarter_emb_dim = 2
        team_emb_dim = 8
        player_emb_dim = 16
        self.position_embedding = nn.Embedding(num_positions, position_emb_dim)
        self.down_emb = nn.Embedding(max_downs, down_emb_dim)
        self.quarter_emb = nn.Embedding(max_quarters, quarter_emb_dim)
        self.team_emb = nn.Embedding(num_teams, team_emb_dim)
        self.player_emb = nn.Embedding(1000, player_emb_dim)

        # Remove feature encoders since we're directly concatenating
        # Note: GNN layers will handle the concatenated features

        self.max_per_graph = 16

        emb_dims = (
            position_emb_dim
            + down_emb_dim
            + quarter_emb_dim
            + team_emb_dim
            + player_emb_dim
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
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

        self.gnn_layers[0] = GATv2Conv(
            5 + 4 + emb_dims,
            hidden_dim,
            edge_dim=3,
            heads=2,
            dropout=dropout,
            concat=False,
        )

        # Frame-level processing
        self.frame_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Historical plays attention
        self.historical_attention = nn.MultiheadAttention(
            hidden_dim * 2,  # bidirectional LSTM output
            num_heads=2,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.max_per_graph),
            nn.GELU(),
            nn.LayerNorm(self.max_per_graph),
            nn.Dropout(dropout),
            nn.Linear(self.max_per_graph, num_route_classes),
        )

    def forward(self, data):
        # Extract batch information
        batch_size = data.batch.max().item() + 1
        device = data.x.device

        # Extract individual features from data.x
        position_idx = data.x[:, 0].long()  # Assuming position index is first column
        raw_features = data.x[
            :, 1:5
        ]  # Rest of features (weight, height, is_offense, motion)

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
                numeric_features,
                player_embedded,
            ],
            dim=1,
        )

        edge_features = data.edge_attr[:, 0:3]  # Keep edge features as is

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, data.edge_index, edge_features)
            node_features = F.gelu(node_features)

        # Separate historical and current plays using plays_elapsed
        historical_mask = data.plays_elapsed > 0
        current_mask = data.plays_elapsed == 0

        # Create new batch indices for historical and current plays separately
        batch_size = data.batch.max().item() + 1
        historical_batch = data.batch[historical_mask]
        current_batch = data.batch[current_mask]

        # Get dense batched representations
        historical_nodes, historical_lens = to_dense_batch(
            node_features[historical_mask],
            historical_batch,
            max_num_nodes=self.max_per_graph,
        )
        current_nodes, current_lens = to_dense_batch(
            node_features[current_mask], current_batch, max_num_nodes=self.max_per_graph
        )

        # Process through LSTM
        historical_encoded, _ = self.frame_lstm(historical_nodes)
        current_encoded, _ = self.frame_lstm(current_nodes)

        # Pool frames to get play-level representations
        # Use the dense masks from to_dense_batch to properly pool
        historical_sum = historical_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)
        current_sum = current_lens.sum(dim=1, keepdim=True).clamp(min=1e-6)

        historical_plays = (
            torch.sum(historical_encoded * historical_lens.unsqueeze(-1), dim=1)
            / historical_sum
        )
        current_plays = (
            torch.sum(current_encoded * current_lens.unsqueeze(-1), dim=1) / current_sum
        )

        # Apply attention between current play and historical plays
        attended_history, _ = self.historical_attention(
            current_plays, historical_plays, historical_plays
        )

        # Get the play-level context as before
        final_representation = torch.cat(
            [current_plays.squeeze(1), attended_history.squeeze(1)], dim=-1
        )  # [batch_size, hidden_dim*4]

        # Get current eligible receivers
        current_eligible_mask = data.eligible_mask & (data.plays_elapsed == 0)
        eligible_nodes = node_features[
            current_eligible_mask
        ]  # [num_eligible_total, hidden_dim]
        eligible_batch_idx = data.batch[current_eligible_mask]  # [num_eligible_total]

        # Get the corresponding play context for each eligible receiver
        play_context = final_representation[
            eligible_batch_idx
        ]  # [num_eligible_total, hidden_dim*4]

        # Combine receiver features with play context
        combined_features = torch.cat(
            [
                eligible_nodes,  # Receiver-specific features
                play_context,  # Play-level context
            ],
            dim=1,
        )  # [num_eligible_total, hidden_dim + hidden_dim*4]

        # Generate unique predictions for each eligible receiver
        route_predictions = self.mlp(
            combined_features
        )  # [num_eligible_total, num_route_classes]

        # print('mlp shape')
        # print(route_predictions.shape)

        # print(route_predictions.isnan().sum())

        return {
            "route_predictions": route_predictions,
            "eligible_batch_idx": eligible_batch_idx,  # Keep track of which batch each prediction belongs to
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
