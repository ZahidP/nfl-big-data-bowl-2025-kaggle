import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_player_embeddings(
    model, player_df, figsize=(15, 10), random_state=42, perplexity=30
):
    """
    Get player embeddings from model, reduce dimensionality, and create visualization.

    Args:
        model: The AnalyzablePlayGNN model
        player_df: DataFrame with player info (must have nflId and displayName_x columns)
        figsize: Tuple specifying figure size
        random_state: Random seed for t-SNE
        perplexity: t-SNE perplexity parameter
    """
    # Get embeddings from model
    embeddings_dict = model.get_player_embeddings()

    # Convert to DataFrame
    embeddings_df = pd.DataFrame(
        [
            {"player_id": int(k), **dict(enumerate(v))}
            for k, v in embeddings_dict.items()
        ]
    )

    # Get embedding columns
    emb_cols = [col for col in embeddings_df.columns if isinstance(col, int)]

    # Scale embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings_df[emb_cols])

    # Apply t-SNE
    print("Fitting TSNE")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(scaled_embeddings)
    print("Fitted TSNE")

    # Add 2D coordinates to DataFrame
    embeddings_df["x"] = embeddings_2d[:, 0]
    embeddings_df["y"] = embeddings_2d[:, 1]

    # Join with player info
    player_df["player_id"] = player_df["nflId"].astype(int)
    result_df = embeddings_df.merge(
        player_df[["player_id", "displayName_x"]], on="player_id", how="inner"
    )

    # Create visualization
    plt.figure(figsize=figsize)

    # Set style
    sns.set_style("whitegrid")

    # Create scatter plot
    scatter = plt.scatter(result_df["x"], result_df["y"], alpha=0.6, s=100)

    # Add labels for each point
    for idx, row in result_df.iterrows():
        plt.annotate(
            row["displayName_x"],
            (row["x"], row["y"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    plt.title("Player Embeddings Visualization (t-SNE)", fontsize=14, pad=20)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return plt.gcf(), result_df


def analyze_clusters(embeddings_df, n_neighbors=5):
    """
    Analyze nearest neighbors in the embedding space.

    Args:
        embeddings_df: DataFrame with embeddings and player info
        n_neighbors: Number of nearest neighbors to find

    Returns:
        DataFrame with nearest neighbors for each player
    """
    from sklearn.neighbors import NearestNeighbors

    # Get embedding coordinates
    X = embeddings_df[["x", "y"]].values

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Create dictionary to store results
    neighbors_dict = {}

    # For each player, get their nearest neighbors
    for idx, player_idx in enumerate(indices):
        player_name = embeddings_df.iloc[idx]["displayName_x"]
        neighbor_names = embeddings_df.iloc[player_idx[1:]]["displayName_x"].values
        neighbor_distances = distances[idx][1:]

        neighbors_dict[player_name] = {
            "nearest_neighbors": list(neighbor_names),
            "distances": list(neighbor_distances),
        }

    return pd.DataFrame.from_dict(neighbors_dict, orient="index")


# Example usage:
"""
# First convert your model to analyzable version
analyzable_model = AnalyzablePlayGNN(*your_model_args)
analyzable_model.load_state_dict(your_model.state_dict())

# Assuming you have player_df with nflId and displayName_x columns
fig, embeddings_df = visualize_player_embeddings(analyzable_model, player_df)

# Save the plot if desired
fig.savefig('player_embeddings.png', dpi=300, bbox_inches='tight')

# Analyze player clusters
neighbors_df = analyze_clusters(embeddings_df)
print(neighbors_df.head())
"""


def analyze_clusters(embeddings_df, n_neighbors=5):
    """
    Analyze nearest neighbors in the embedding space.

    Args:
        embeddings_df: DataFrame with embeddings and player info
        n_neighbors: Number of nearest neighbors to find

    Returns:
        DataFrame with nearest neighbors for each player
    """
    from sklearn.neighbors import NearestNeighbors

    # Get embedding coordinates
    X = embeddings_df[["x", "y"]].values

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Create dictionary to store results
    neighbors_dict = {}

    # For each player, get their nearest neighbors
    for idx, player_idx in enumerate(indices):
        player_name = embeddings_df.iloc[idx]["displayName_x"]
        neighbor_names = embeddings_df.iloc[player_idx[1:]]["displayName_x"].values
        neighbor_distances = distances[idx][1:]

        neighbors_dict[player_name] = {
            "nearest_neighbors": list(neighbor_names),
            "distances": list(neighbor_distances),
        }

    return pd.DataFrame.from_dict(neighbors_dict, orient="index")
