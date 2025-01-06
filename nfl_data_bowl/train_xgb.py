import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import top_k_accuracy_score
import pandas as pd
from scipy.stats import entropy
import xgboost as xgb
from sklearn.metrics import classification_report
from IPython.display import display, HTML
from typing import Optional, Literal, Union, Dict


def prepare_route_prediction_data(
    merged_df, feature_encoders=None, training=True, scaler=None, do_scaling=True
):
    """
    Joins the necessary tables and prepares features for route prediction

    Args:
        merged_df: DataFrame containing the input data
        feature_encoders: Optional dict of pre-fitted encoders for inference
        training: Boolean indicating if this is for training (True) or inference (False)
        scaler: Optional pre-fitted StandardScaler
        do_scaling: Boolean indicating whether to perform feature scaling

    Returns:
        X: Feature matrix
        y: Target labels (None if training=False)
        feature_encoders: Dictionary of fitted encoders (only if training=True)
        scaler: Fitted StandardScaler (if do_scaling=True)
    """
    # Create a copy and handle NaN routes based on mode
    merged_df = merged_df.copy()
    if training:
        merged_df = merged_df.dropna(subset=["routeRan"])
        if len(merged_df) == 0:
            raise ValueError("No data remaining after removing NaN routes")
        print(f"Number of samples after removing NaN routes: {len(merged_df)}")

    merged_df["minutes_remaining"] = merged_df["gameClock"]

    merged_df["score_diff"] = np.abs(
        merged_df["preSnapHomeScore"] - merged_df["preSnapVisitorScore"]
    )

    # Define feature groups
    numerical_features = [
        "yardsToGo",
        "yardlineNumber",
        "minutes_remaining",
        "dis_wr_1",
        "dis_wr_2",
        # "dis_wr_3",
        "height_inches",
        "weight",
        "distance_from_sideline",
        "distance_from_los",
        # "dis_wr_4",
        # "dis_wr_5",
        "wr_center_distance",
        "score_diff",
    ]

    categorical_features = [
        "possessionTeam",
        "defensiveTeam",
        "yardlineSide",
        "offenseFormation",
        "receiverAlignment",
        "down",
        "position",
        "quarter",
    ]

    boolean_features = ["inMotionAtBallSnap"]

    if training:  #  and not encoder
        # Initialize dictionary to store encoders
        feature_encoders = {}

        # Fit and transform categorical features
        for feature in categorical_features:
            encoder = LabelEncoder()
            merged_df[feature + "_encoded"] = encoder.fit_transform(merged_df[feature])
            feature_encoders[feature] = encoder
    else:
        # Validate that we have all necessary encoders
        if feature_encoders is None:
            raise ValueError("feature_encoders must be provided when training=False")

        required_encoders = set(categorical_features + ["routeRan"])
        missing_encoders = required_encoders - set(feature_encoders.keys())
        if missing_encoders:
            raise ValueError(f"Missing encoders for features: {missing_encoders}")

        # Transform categorical features using provided encoders
        for feature in categorical_features:
            try:
                merged_df[feature + "_encoded"] = feature_encoders[feature].transform(
                    merged_df[feature]
                )
            except ValueError as e:
                print(
                    f"Error encoding feature {feature}. This might be due to new categories."
                )
                raise e

    # Create feature matrix X
    encoded_categorical_features = [f + "_encoded" for f in categorical_features]
    X = merged_df[numerical_features + encoded_categorical_features + boolean_features]

    # Handle scaling
    if do_scaling:
        if scaler is None and not training:
            raise ValueError(
                "scaler must be provided for inference when do_scaling=True"
            )
        elif training:
            if scaler is None:
                scaler = StandardScaler()
                X = pd.DataFrame(
                    scaler.fit_transform(X), columns=X.columns, index=X.index
                )
            else:
                X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        else:  # inference mode with provided scaler
            X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    # Handle target variable based on mode
    if training:
        route_encoder = LabelEncoder()
        routes = merged_df["routeRan"].fillna("UNKNOWN")
        y = route_encoder.fit_transform(routes)
        feature_encoders["routeRan"] = route_encoder

        # Verify no NaNs in encoded target
        if np.isnan(y).any():
            raise ValueError("Found NaN values in encoded target variable")

        print(f"Number of unique routes: {len(np.unique(y))}")
        print("Route distribution:")
        for label in np.unique(y):
            route_name = route_encoder.inverse_transform([label])[0]
            count = (y == label).sum()
            print(f"{route_name}: {count} samples")
    else:
        y = None
        if "routeRan" in merged_df.columns:
            try:
                y = feature_encoders["routeRan"].transform(
                    merged_df["routeRan"].fillna("UNKNOWN")
                )
            except:
                print("Warning: Could not encode some route labels during inference")

    return X, y, (feature_encoders if training else None), scaler


def create_data_splits(
    merged_df: pd.DataFrame,
    split_method: Literal[
        "week", "random", "stratified_by_game", "stratified_by_play"
    ] = "week",
    train_size: Union[list, float] = 0.7,
    val_size: Union[list, float] = 0.15,
    test_size: Union[list, float] = 0.15,
    week_col: str = "week",
    game_id_col: str = "gameId",
    play_id_col: str = "playId",
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Creates train/val/test splits based on different sampling methods.

    Args:
        merged_df: Input DataFrame
        split_method: One of "week", "random", "stratified_by_game", or "stratified_by_play"
        train_size: List of weeks or float for random sampling
        val_size: List of weeks or float for random sampling
        test_size: List of weeks or float for random sampling
        week_col: Name of week column
        game_id_col: Name of game ID column
        play_id_col: Name of play ID column
        random_state: Random seed

    Returns:
        Dictionary containing train, validation and test DataFrames
    """
    if split_method == "week":
        if not all(isinstance(x, list) for x in [train_size, val_size, test_size]):
            raise ValueError(
                "For week-based splitting, sizes must be lists of week numbers"
            )

        train_mask = merged_df[week_col].isin(train_size)
        val_mask = merged_df[week_col].isin(val_size)
        test_mask = merged_df[week_col].isin(test_size)

    elif split_method == "random":
        if not all(isinstance(x, float) for x in [train_size, val_size, test_size]):
            raise ValueError("For random splitting, sizes must be float fractions")

        if not np.isclose(sum([train_size, val_size, test_size]), 1.0):
            raise ValueError("Split proportions must sum to 1.0")

        train_df, temp_df = train_test_split(
            merged_df,
            train_size=train_size,
            random_state=random_state,
            stratify=merged_df.routeRan,
        )

        relative_val_size = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val_size,
            random_state=random_state,
            stratify=temp_df.routeRan,
        )

        return {"train": train_df, "val": val_df, "test": test_df}

    elif split_method == "stratified_by_game":
        if not all(isinstance(x, float) for x in [train_size, val_size, test_size]):
            raise ValueError(
                "For game-stratified splitting, sizes must be float fractions"
            )

        # Get unique game IDs
        game_ids = merged_df[game_id_col].unique()

        # Split game IDs into train/val/test
        train_games, temp_games = train_test_split(
            game_ids, train_size=train_size, random_state=random_state
        )

        relative_val_size = val_size / (val_size + test_size)
        val_games, test_games = train_test_split(
            temp_games, train_size=relative_val_size, random_state=random_state
        )

        # Create masks based on game IDs
        train_mask = merged_df[game_id_col].isin(train_games)
        val_mask = merged_df[game_id_col].isin(val_games)
        test_mask = merged_df[game_id_col].isin(test_games)

    elif split_method == "stratified_by_play":
        if not all(isinstance(x, float) for x in [train_size, val_size, test_size]):
            raise ValueError(
                "For play-stratified splitting, sizes must be float fractions"
            )

        # Create unique game-play combinations
        merged_df["game_play"] = (
            merged_df[game_id_col].astype(str)
            + "_"
            + merged_df[play_id_col].astype(str)
        )
        unique_plays = merged_df["game_play"].unique()

        # Split unique plays into train/val/test
        train_plays, temp_plays = train_test_split(
            unique_plays, train_size=train_size, random_state=random_state
        )

        relative_val_size = val_size / (val_size + test_size)
        val_plays, test_plays = train_test_split(
            temp_plays, train_size=relative_val_size, random_state=random_state
        )

        # Create masks based on game-play combinations
        train_mask = merged_df["game_play"].isin(train_plays)
        val_mask = merged_df["game_play"].isin(val_plays)
        test_mask = merged_df["game_play"].isin(test_plays)

        # Remove temporary column
        merged_df = merged_df.drop("game_play", axis=1)

    else:
        raise ValueError(f"Unknown split method: {split_method}")

    if split_method != "random":
        return {
            "train": merged_df[train_mask],
            "val": merged_df[val_mask],
            "test": merged_df[test_mask],
        }


def train_route_prediction_pipeline(
    merged_df: pd.DataFrame,
    split_method: Literal[
        "week", "random", "stratified_by_game", "stratified_by_play"
    ] = "week",
    train_split: Union[list, float] = 0.7,
    val_split: Union[list, float] = 0.15,
    test_split: Union[list, float] = 0.15,
    max_depth: int = 10,
    week_col: str = "week",
    time_col: str = "time",
    game_id_col: str = "gameId",
    play_id_col: str = "playId",
    random_state: int = 42,
):
    """
    Complete pipeline for route prediction including data preparation,
    feature engineering, and model training with flexible sampling methods.

    Parameters:
    - merged_df: DataFrame containing all required columns
    - split_method: Method to use for splitting data
    - train_split: List of weeks or fraction for training
    - val_split: List of weeks or fraction for validation
    - test_split: List of weeks or fraction for testing
    - max_depth: Maximum depth for XGBoost model
    - week_col: Name of the week column
    - time_col: Name of the time column
    - game_id_col: Name of the game ID column
    - play_id_col: Name of the play ID column
    - random_state: Random seed

    Returns:
    - model: Trained XGBoost model
    - metrics: Dictionary of evaluation metrics
    - feature_encoders: Dictionary of fitted encoders
    - split_info: Information about the data splits
    - scaler: Fitted StandardScaler
    """
    print(f"Initial data shape: {merged_df.shape}")

    # Step 1: Create data splits
    splits = create_data_splits(
        merged_df=merged_df,
        split_method=split_method,
        train_size=train_split,
        val_size=val_split,
        test_size=test_split,
        week_col=week_col,
        game_id_col=game_id_col,
        play_id_col=play_id_col,
        random_state=random_state,
    )

    # Step 2: Prepare features for training data
    X_train, y_train, feature_encoders, _ = prepare_route_prediction_data(
        merged_df, training=True, do_scaling=False
    )

    X_train, y_train, _, _ = prepare_route_prediction_data(
        splits["train"],
        training=False,
        do_scaling=False,
        feature_encoders=feature_encoders,
    )

    # Step 3: Prepare validation and test data using the same encoders
    X_val, y_val, _, _ = prepare_route_prediction_data(
        splits["val"],
        feature_encoders=feature_encoders,
        training=False,
        do_scaling=False,
    )

    X_test, y_test, _, _ = prepare_route_prediction_data(
        splits["test"],
        feature_encoders=feature_encoders,
        training=False,
        do_scaling=False,
    )

    # Print split information
    print("\nData split sizes:")
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    if split_method in ["stratified_by_game", "stratified_by_play"]:
        print("\nUnique games in each split:")
        print(f"Train: {splits['train'][game_id_col].nunique()} games")
        print(f"Validation: {splits['val'][game_id_col].nunique()} games")
        print(f"Test: {splits['test'][game_id_col].nunique()} games")

        print("\nUnique plays in each split:")
        train_plays = set(
            zip(splits["train"][game_id_col], splits["train"][play_id_col])
        )
        val_plays = set(zip(splits["val"][game_id_col], splits["val"][play_id_col]))
        test_plays = set(zip(splits["test"][game_id_col], splits["test"][play_id_col]))

        print(f"Train: {len(train_plays)} plays")
        print(f"Validation: {len(val_plays)} plays")
        print(f"Test: {len(test_plays)} plays")

        # Verify no leakage
        if len(train_plays.intersection(val_plays)) > 0:
            raise ValueError("Play ID leakage detected between train and val!")
        if len(train_plays.intersection(test_plays)) > 0:
            raise ValueError("Play ID leakage detected between train and test!")
        if len(val_plays.intersection(test_plays)) > 0:
            raise ValueError("Play ID leakage detected between val and test!")

    # Step 4: Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Store scaler in encoders
    feature_encoders["scaler"] = scaler

    # Create split info dictionary
    split_info = {
        "train_index": X_train.index,
        "val_index": X_val.index,
        "test_index": X_test.index,
        "feature_names": X_train.columns.tolist(),
        "split_method": split_method,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    }

    if split_method == "week":
        split_info["week_ranges"] = {
            "train": (min(train_split), max(train_split)),
            "val": (min(val_split), max(val_split)),
            "test": (min(test_split), max(test_split)),
        }

    # Train model
    model = xgb.XGBClassifier(
        num_class=len(np.unique(y_train)),
        # learning_rate=0.05,
        max_depth=max_depth,
        n_estimators=200,
        use_label_encoder=False,
        objective="multi:softmax",
        early_stopping_rounds=50,
        random_state=random_state,
    )

    model.fit(
        X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True
    )

    # Generate predictions and calculate metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    splits["y_test"] = y_test

    # Add feature importance information
    feature_importance = pd.DataFrame(
        {"feature": X_test.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    metrics = classification_report(y_test, y_pred, output_dict=True)
    # Calculate basic metrics
    # metrics = classification_report(splits["y_test"], y_pred, output_dict=True)

    metrics["feature_importance"] = feature_importance
    metrics["top_2_accuracy"] = top_k_accuracy_score(
        splits["y_test"], y_pred_proba, k=2
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        splits["y_test"], y_pred_proba, k=3
    )

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs)
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy
        sorted_probs = np.sort(probabilities, axis=1)
        entropy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

        return {
            "mean_entropy": np.mean(prediction_entropies),
            "median_entropy": np.median(prediction_entropies),
            "mean_normalized_entropy": np.mean(normalized_entropies),
            "mean_entropy_margin": np.mean(entropy_margin),
            "mean_log_likelihood": np.mean(log_likelihood),
            "median_log_likelihood": np.median(log_likelihood),
        }

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(
        y_pred_proba, splits["y_test"]
    )

    # New formatted output function using pandas DataFrames
    def display_formatted_metrics(metrics_dict):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": feature_encoders["routeRan"].inverse_transform(
                            [int(class_label)]
                        )[0],
                        # 'Actual Count': metrics_dict[class_label]['actual_count'],
                        # 'Predicted Count': metrics_dict[class_label]['predicted_count'],
                        "Precision": f"{metrics_dict[class_label]['precision']:.3f}",
                        "Recall": f"{metrics_dict[class_label]['recall']:.3f}",
                        "F1-score": f"{metrics_dict[class_label]['f1-score']:.3f}",
                        "Support": metrics_dict[class_label]["support"],
                    }
                )
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        class_df = pd.DataFrame(class_metrics)

        # Top-K accuracy table
        topk_metrics = pd.DataFrame(
            [
                {
                    "Metric": "Accuracy Type",
                    "Top-1 (standard)": f"{metrics_dict['accuracy']:.3f}",
                    "Top-2": f"{metrics_dict['top_2_accuracy']:.3f}",
                    "Top-3": f"{metrics_dict['top_3_accuracy']:.3f}",
                }
            ]
        )

        # Uncertainty metrics table
        uncertainty_df = pd.DataFrame(
            [
                {
                    "Mean Entropy": f"{metrics_dict['uncertainty_metrics']['mean_entropy']:.3f}",
                    "Median Entropy": f"{metrics_dict['uncertainty_metrics']['median_entropy']:.3f}",
                    "Mean Normalized Entropy": f"{metrics_dict['uncertainty_metrics']['mean_normalized_entropy']:.3f}",
                    "Mean Entropy Margin": f"{metrics_dict['uncertainty_metrics']['mean_entropy_margin']:.3f}",
                    "Mean Log Likelihood": f"{metrics_dict['uncertainty_metrics']['mean_log_likelihood']:.3f}",
                    "Median Log Likelihood": f"{metrics_dict['uncertainty_metrics']['median_log_likelihood']:.3f}",
                }
            ]
        )

        # Overall metrics table
        overall_metrics = []
        for avg_type in ["macro avg", "weighted avg"]:
            metrics_row = {"Average Type": avg_type}
            metrics_row.update(
                {k: f"{v:.3f}" for k, v in metrics_dict[avg_type].items()}
            )
            overall_metrics.append(metrics_row)
        overall_df = pd.DataFrame(overall_metrics)

        # Display tables with styling
        print("\n=== Per-Class Performance ===")
        display(
            HTML(
                class_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Top-K Accuracy ===")
        display(
            HTML(
                topk_metrics.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Uncertainty Metrics ===")
        display(
            HTML(
                uncertainty_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

        print("\n=== Overall Metrics ===")
        display(
            HTML(
                overall_df.style.set_properties(
                    **{"text-align": "center", "padding": "8px"}
                ).to_html()
            )
        )

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics)

    return model, metrics, feature_encoders, split_info, scaler
