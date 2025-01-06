import torch
import torch.nn as nn
import numpy as np
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from torch_geometric.utils import to_dense_batch

from nfl_data_bowl.models.gnn_historical_enhanced import PlayGNN


class AnalyzablePlayGNN(PlayGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_player_embeddings(self):
        """
        Returns the learned player embeddings from the model.

        Returns:
            dict: Dictionary mapping player IDs to their embeddings
        """
        embeddings = self.player_emb.weight.detach().cpu().numpy()
        return {
            player_id: embeddings[player_id]
            for player_id in range(len(embeddings))
            if player_id < 1000  # Only return actual player embeddings
        }

    def compute_feature_importance(self, data, target_class=None):
        """
        Compute feature importance scores using integrated gradients.

        Args:
            data: A batch of input data
            target_class: Optional specific route class to analyze

        Returns:
            dict: Dictionary containing importance scores for different feature groups
        """
        self.eval()

        # Store original input
        original_x = data.x.clone()

        # Create baseline (zero embeddings)
        baseline_x = torch.zeros_like(data.x)

        def forward_func(x):
            # Temporarily replace input features
            data.x = x
            output = self.forward(data)
            if target_class is not None:
                return output["route_predictions"][:, target_class]
            return output["route_predictions"].sum(dim=1)

        # Initialize Integrated Gradients
        ig = IntegratedGradients(forward_func)

        # Compute attributions
        attributions = ig.attribute(original_x, baseline_x, n_steps=50)

        # Aggregate importance scores by feature groups
        importance_scores = {
            "position": attributions[:, 0].abs().mean().item(),
            "player_attributes": attributions[:, 1:5].abs().mean().item(),
            "game_situation": attributions[:, 5:].abs().mean().item(),
        }

        # Restore original input
        data.x = original_x

        return importance_scores

    def analyze_attention_patterns(self, data):
        """
        Analyze the attention patterns in the historical attention layer.

        Args:
            data: A batch of input data

        Returns:
            torch.Tensor: Attention weights for historical plays
        """
        self.eval()
        with torch.no_grad():
            # Forward pass to get attention weights
            node_features = self.forward_gnn_layers(data)

            # Get historical and current plays
            historical_mask = data.plays_elapsed > 0
            current_mask = data.plays_elapsed == 0

            historical_batch = data.batch[historical_mask]
            current_batch = data.batch[current_mask]

            # Get dense representations
            historical_nodes, _ = to_dense_batch(
                node_features[historical_mask],
                historical_batch,
                max_num_nodes=self.max_per_graph,
            )
            current_nodes, _ = to_dense_batch(
                node_features[current_mask],
                current_batch,
                max_num_nodes=self.max_per_graph,
            )

            # Process through LSTM
            historical_encoded, _ = self.frame_lstm(historical_nodes)
            current_encoded, _ = self.frame_lstm(current_nodes)

            # Get attention weights
            _, attention_weights = self.historical_attention(
                current_encoded,
                historical_encoded,
                historical_encoded,
                need_weights=True,
            )

            return attention_weights

    def forward_gnn_layers(self, data):
        """Helper method to run forward pass through GNN layers only"""
        # Extract and process features as in the original forward method
        position_idx = data.x[:, 0].long()
        raw_features = data.x[:, 1:5]

        position_embedded = self.position_embedding(position_idx)
        down_embedded = self.down_emb(data.down[data.batch] - 1)
        quarter_embedded = self.quarter_emb(data.quarter[data.batch] - 1)
        team_embedded = self.team_emb(data.offense_team[data.batch])
        player_embedded = self.player_emb(data.player_ids)

        numeric_features = torch.cat(
            [
                data.game_clock[data.batch],
                data.yardline.unsqueeze(1)[data.batch],
                data.yards_to_go.unsqueeze(1)[data.batch],
                data.offense_score.unsqueeze(1)[data.batch],
                data.defense_score.unsqueeze(1)[data.batch],
            ],
            dim=1,
        )

        node_features = torch.cat(
            [
                position_embedded,
                raw_features,
                down_embedded,
                quarter_embedded,
                team_embedded,
                numeric_features,
                player_embedded,
            ],
            dim=1,
        )

        # Run through GNN layers
        edge_features = data.edge_attr[:, 0:3]
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, data.edge_index, edge_features)
            node_features = F.gelu(node_features)

        return node_features


# Example usage:
def analyze_model(model, data_batch):
    """
    Perform comprehensive analysis of the model.

    Args:
        model: An instance of AnalyzablePlayGNN
        data_batch: A batch of input data

    Returns:
        dict: Dictionary containing various analysis results
    """
    # Get player embeddings
    player_embeddings = model.get_player_embeddings()

    # Compute feature importance for each route class
    feature_importance_by_class = {}
    for route_class in range(model.mlp[-1].out_features):
        importance_scores = model.compute_feature_importance(
            data_batch, target_class=route_class
        )
        feature_importance_by_class[f"route_{route_class}"] = importance_scores

    # Analyze attention patterns
    attention_weights = model.analyze_attention_patterns(data_batch)

    return {
        "player_embeddings": player_embeddings,
        "feature_importance_by_class": feature_importance_by_class,
        "attention_weights": attention_weights,
    }
