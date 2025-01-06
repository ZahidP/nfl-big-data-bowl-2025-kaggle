import torch
import numpy as np
from sklearn.metrics import classification_report, top_k_accuracy_score
import pandas as pd
import torch
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import entropy
from IPython.display import display, HTML
from nfl_data_bowl.train_xgb import prepare_route_prediction_data
from nfl_data_bowl.data_utils.data_common import filter_by_game_play_ids


def get_xgb_preds(xgb_model, plays_df, batch, feature_encoders, scaler):

    # Get corresponding plays for XGBoost
    ids = list(
        zip(
            batch.game_id[batch.eligible_mask].cpu().tolist(),
            batch.play_id[batch.eligible_mask].cpu().tolist(),
            batch.player_ids[batch.eligible_mask].cpu().tolist(),
        )
    )
    batch_df = filter_by_game_play_ids(plays_df, ids)

    # XGB predictions
    X, y, _, _2 = prepare_route_prediction_data(
        batch_df,
        training=False,
        feature_encoders=feature_encoders,
        scaler=scaler,
    )
    xgb_probs = xgb_model.predict_proba(X)
    return xgb_probs


def evaluate_route_predictions_table(
    model,
    dataloader_,
    dataset_,
    device="cuda",
    entropy_ceiling=3,
    key="route_predictions",
    softmax=False,
    plays_df=None,
    feature_encoders=None,
    scaler=None,
):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics and formatted table output.
    Now includes play-level and field position metrics.

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    try:
        model.eval()
    except Exception as e:
        print("Trying XGB")
    all_predictions = []
    all_probabilities = []
    all_targets = []
    all_metadata = []  # New list to store metadata

    with torch.no_grad():
        for batch in dataloader_:
            try:
                batch = batch.to(device)
                targets = batch.route_targets[batch.eligible_mask]
                try:
                    output = model(batch)
                    probabilities = output[key]

                except Exception as e:
                    probabilities = get_xgb_preds(
                        model, plays_df, batch, feature_encoders, scaler
                    )
                    output = {}

                if "target" in output.keys():
                    torch.testing.assert_close(targets, torch.tensor(output["target"]))

                if isinstance(probabilities, torch.Tensor):
                    if softmax:
                        probabilities = torch.softmax(probabilities, dim=1)
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = (
                        torch.tensor(probabilities).argmax(dim=1).cpu().numpy()
                    )

                # Extract metadata for eligible players
                metadata = {
                    "play_id": batch.play_id[batch.eligible_mask].cpu().numpy(),
                    "game_id": batch.game_id[batch.eligible_mask].cpu().numpy(),
                    "player_id": batch.player_ids[batch.eligible_mask].cpu().numpy(),
                    "yardline": batch.yardline[batch.eligible_mask].cpu().numpy(),
                    "down": (
                        batch.down[batch.eligible_mask].cpu().numpy()
                        if hasattr(batch, "down")
                        else None
                    ),
                    "yards_to_go": (
                        batch.yards_to_go[batch.eligible_mask].cpu().numpy()
                        if hasattr(batch, "yards_to_go")
                        else None
                    ),
                }

                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
                all_metadata.append(metadata)
            except Exception as e:
                raise e

    try:
        y_pred = np.concatenate(all_predictions)
        y_pred_proba = np.concatenate(all_probabilities)
        y_test = np.concatenate(all_targets)

        # Concatenate metadata
        combined_metadata = {
            key: np.concatenate([batch[key] for batch in all_metadata])
            for key in all_metadata[0].keys()
            if all_metadata[0][key] is not None
        }

    except Exception as e:
        print(all_predictions)
        raise e

    # Apply entropy filtering
    prediction_entropies = np.apply_along_axis(entropy, 1, y_pred_proba)
    entropy_subset = prediction_entropies < entropy_ceiling

    y_pred = y_pred[entropy_subset]
    y_pred_proba = y_pred_proba[entropy_subset]
    y_test = y_test[entropy_subset]

    # Also filter metadata
    filtered_metadata = {
        key: value[entropy_subset] for key, value in combined_metadata.items()
    }

    # Create DataFrame with all predictions and metadata
    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "probabilities": y_pred_proba.tolist(),
            "predicted": y_pred,
            "entropy": prediction_entropies[entropy_subset],
            **filtered_metadata,
        }
    )

    # Convert route indices to names
    predictions_df["actual_route"] = predictions_df["actual"].map(dataset_.idx_to_route)
    predictions_df["predicted_route"] = predictions_df["predicted"].map(
        dataset_.idx_to_route
    )

    # Calculate base metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    # Calculate actual vs predicted counts
    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs + 1e-10)
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

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # # Calculate Top-K accuracy
    # def top_k_accuracy(y_true, y_pred_proba, k):
    #     top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]

    #     correct = 0
    #     for i, true_label in enumerate(y_true):
    #         if true_label in top_k_predictions[i]:
    #             correct += 1
    #     return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=2, labels=list(range(0, 13))
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=3, labels=list(range(0, 13))
    )

    # Add predictions DataFrame to metrics
    metrics["predictions_df"] = predictions_df

    # Add some basic grouped metrics
    def calculate_grouped_metrics(df):
        play_metrics = {}

        # Redzone analysis (inside 20 yard line)
        redzone_mask = df["yardline"] <= 20
        play_metrics["redzone"] = {
            "accuracy": (
                df[redzone_mask]["actual"] == df[redzone_mask]["predicted"]
            ).mean(),
            "sample_size": redzone_mask.sum(),
            "route_distribution": df[redzone_mask]["actual_route"]
            .value_counts()
            .to_dict(),
        }

        # Play-level accuracy
        play_level_accuracy = (
            df.groupby(["play_id", "game_id"])
            .apply(lambda x: (x["actual"] == x["predicted"]).mean())
            .describe()
            .to_dict()
        )
        play_metrics["play_level"] = play_level_accuracy

        # Most common route combinations
        play_metrics["route_combinations"] = (
            df.groupby(["play_id", "game_id"])["actual_route"]
            .agg(
                lambda x: tuple(sorted(x))
            )  # Convert to sorted tuple for consistent ordering
            .value_counts()
            .head(10)
            .to_dict()
        )

        return play_metrics

    def display_formatted_metrics(metrics_dict, dataset_):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": dataset_.idx_to_route[int(class_label)],
                        "Actual Count": metrics_dict[class_label]["actual_count"],
                        "Predicted Count": metrics_dict[class_label]["predicted_count"],
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

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics, dataset_)

    metrics["grouped_metrics"] = calculate_grouped_metrics(predictions_df)

    return metrics, (y_test, y_pred, y_pred_proba, predictions_df)


def evaluate_route_predictions_table_old(
    model,
    dataloader_,
    dataset_,
    device="cuda",
    entropy_ceiling=3,
    key="route_predictions",
    softmax=False,
):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics and formatted table output

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    # [Previous code for model evaluation remains the same until formatting section]
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_:
            try:
                # print(batch)
                batch = batch.to(device)
                output = model(batch)
                probabilities = output[key]

                targets = batch.route_targets[batch.eligible_mask]

                if "target" in output.keys():
                    torch.testing.assert_close(targets, torch.tensor(output["target"]))

                if isinstance(probabilities, torch.Tensor):
                    if softmax:
                        probabilities = torch.softmax(probabilities, dim=1)
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = (
                        torch.tensor(probabilities).argmax(dim=1).cpu().numpy()
                    )

                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                raise e
    try:
        y_pred = np.concatenate(all_predictions)
        y_pred_proba = np.concatenate(all_probabilities)
        y_test = np.concatenate(all_targets)
    except Exception as e:
        print(all_predictions)
        raise e

    prediction_entropies = np.apply_along_axis(entropy, 1, y_pred_proba)
    entropy_subset = prediction_entropies < entropy_ceiling

    y_pred = y_pred[entropy_subset]
    y_pred_proba = y_pred_proba[entropy_subset]
    y_test = y_test[entropy_subset]

    metrics = classification_report(y_test, y_pred, output_dict=True)
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    # Calculate actual vs predicted counts
    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(true_label_probs + 1e-10)
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

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # # Calculate Top-K accuracy
    # def top_k_accuracy(y_true, y_pred_proba, k):
    #     top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]

    #     correct = 0
    #     for i, true_label in enumerate(y_true):
    #         if true_label in top_k_predictions[i]:
    #             correct += 1
    #     return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=2, labels=list(range(0, 13))
    )
    metrics["top_3_accuracy"] = top_k_accuracy_score(
        y_test, y_pred_proba, k=3, labels=list(range(0, 13))
    )

    # New formatted output function using pandas DataFrames
    def display_formatted_metrics(metrics_dict, dataset_):
        # Per-class metrics table
        class_metrics = []
        for class_label in sorted([k for k in metrics_dict.keys() if k.isdigit()]):
            try:
                class_metrics.append(
                    {
                        "Route": dataset_.idx_to_route[int(class_label)],
                        "Actual Count": metrics_dict[class_label]["actual_count"],
                        "Predicted Count": metrics_dict[class_label]["predicted_count"],
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

    metrics["display_tables"] = lambda: display_formatted_metrics(metrics, dataset_)
    metrics["prediction_entropies"] = prediction_entropies

    return metrics, (y_test, y_pred, y_pred_proba)


# Usage example:
"""
# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
metrics, (y_test, y_pred, y_pred_proba) = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

# Display formatted tables
metrics['display_tables']()
"""


def evaluate_route_predictions(model, dataloader_, dataset_, device="cuda"):
    """
    Evaluates a PyTorch route prediction model with enhanced metrics

    Parameters:
    - model: PyTorch model
    - dataloader: PyTorch DataLoader for evaluation
    - dataset: MultiRoutePlayDataset instance
    - device: Device to run evaluation on
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    prev = 0

    with torch.no_grad():
        for batch in dataloader_:
            batch = batch.to(device)
            output = model(batch)
            predictions = output["route_predictions"]

            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(predictions, dim=1)

            # Print average time for this batch
            batch_time = (
                batch.time.float().mean().item()
            )  # Assuming 'time' is an attribute in your batch
            batch_min = batch.time.float().min().item()
            batch_max = batch.time.float().max().item()
            print(
                f"Batch average time diff from prev: {batch_time - prev:.3f}, batch min: {batch_min}, batch_max: {batch_max}"
            )
            prev = batch_time

            # Get targets for eligible receivers
            targets = batch.route_targets[batch.eligible_mask]

            # Move to CPU and convert to numpy
            all_predictions.append(predictions.argmax(dim=1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_predictions)
    y_pred_proba = np.concatenate(all_probabilities)
    y_test = np.concatenate(all_targets)

    # Calculate basic metrics
    metrics = classification_report(y_test, y_pred, output_dict=True)

    # Calculate actual vs predicted counts
    actuals_predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    for class_label in np.unique(y_test):
        actual_count = len(
            actuals_predictions[actuals_predictions["Actual"] == class_label]
        )
        predicted_count = len(
            actuals_predictions[actuals_predictions["Predicted"] == class_label]
        )
        metrics[str(class_label)]["actual_count"] = actual_count
        metrics[str(class_label)]["predicted_count"] = predicted_count

    # Calculate uncertainty metrics
    def calculate_uncertainty_metrics(probabilities, y_true):
        # Calculate entropy for each prediction
        prediction_entropies = np.apply_along_axis(entropy, 1, probabilities)

        # Calculate log likelihood of true labels
        true_label_probs = np.array(
            [prob[true] for prob, true in zip(probabilities, y_true)]
        )
        log_likelihood = np.log(
            true_label_probs + 1e-10
        )  # Add small epsilon to prevent log(0)

        # Calculate normalized entropy
        max_entropy = np.log2(probabilities.shape[1])
        normalized_entropies = prediction_entropies / max_entropy

        # Calculate entropy margin
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

    metrics["uncertainty_metrics"] = calculate_uncertainty_metrics(y_pred_proba, y_test)

    # Calculate Top-K accuracy
    def top_k_accuracy(y_true, y_pred_proba, k):
        top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_predictions[i]:
                correct += 1
        return correct / len(y_true)

    metrics["top_2_accuracy"] = top_k_accuracy(y_test, y_pred_proba, 2)
    metrics["top_3_accuracy"] = top_k_accuracy(y_test, y_pred_proba, 3)

    # Enhanced formatting function using dataset's route mapping
    def format_metrics(metrics_dict, dataset_):
        formatted_report = "\nClassification Report:\n-------------------"

        # Print individual class metrics
        for class_label in sorted(
            [
                k
                for k in metrics_dict.keys()
                if k
                not in [
                    "uncertainty_metrics",
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                    "top_2_accuracy",
                    "top_3_accuracy",
                    "confidence_metrics",
                ]
            ]
        ):

            try:
                if class_label.isdigit():
                    class_metrics = metrics_dict[class_label]
                    route_name = dataset_.idx_to_route[int(class_label)]
                    formatted_report += f"\n\nClass {route_name}:\n"
                    formatted_report += (
                        f"    Actual count: {class_metrics['actual_count']}\n"
                    )
                    formatted_report += (
                        f"    Predicted count: {class_metrics['predicted_count']}\n"
                    )
                    formatted_report += (
                        f"    Precision: {class_metrics['precision']:.3f}\n"
                    )
                    formatted_report += f"    Recall: {class_metrics['recall']:.3f}\n"
                    formatted_report += (
                        f"    F1-score: {class_metrics['f1-score']:.3f}\n"
                    )
                    formatted_report += f"    Support: {class_metrics['support']}"
            except Exception as e:
                print(f"Could not get metrics for {class_label}")

        # Print Top-K metrics
        formatted_report += "\n\nTop-K Accuracy:\n--------------"
        formatted_report += (
            f"\nTop-1 (standard) accuracy: {metrics_dict['accuracy']:.3f}"
        )
        formatted_report += f"\nTop-2 accuracy: {metrics_dict['top_2_accuracy']:.3f}"
        formatted_report += f"\nTop-3 accuracy: {metrics_dict['top_3_accuracy']:.3f}"

        # Print uncertainty metrics
        uncertainty_metrics = metrics_dict["uncertainty_metrics"]
        formatted_report += "\n\nUncertainty Metrics:\n-------------------"
        formatted_report += (
            f"\nMean prediction entropy: {uncertainty_metrics['mean_entropy']:.3f}"
        )
        formatted_report += (
            f"\nMedian prediction entropy: {uncertainty_metrics['median_entropy']:.3f}"
        )
        formatted_report += f"\nMean normalized entropy: {uncertainty_metrics['mean_normalized_entropy']:.3f}"
        formatted_report += (
            f"\nMean entropy margin: {uncertainty_metrics['mean_entropy_margin']:.3f}"
        )
        formatted_report += (
            f"\nMean log likelihood: {uncertainty_metrics['mean_log_likelihood']:.3f}"
        )
        formatted_report += f"\nMedian log likelihood: {uncertainty_metrics['median_log_likelihood']:.3f}"

        # Print overall metrics
        formatted_report += "\n\nOverall Metrics:\n---------------"
        for avg_type in ["macro avg", "weighted avg"]:
            formatted_report += f"\n{avg_type}:"
            for metric, value in metrics_dict[avg_type].items():
                formatted_report += f"\n    {metric}: {value:.3f}"

        return formatted_report

    metrics["format_report"] = lambda: format_metrics(metrics, dataset_)

    return metrics, (y_test, y_pred, y_pred_proba)


# Usage example:
"""
# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate model
metrics, (y_test, y_pred, y_pred_proba) = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

# Print formatted report
print(metrics['format_report']())
"""


def analyze_distributional_drift(train_loader, test_loader, num_batches=None):
    """
    Analyze distributional drift between train and test sets.

    Parameters:
    - train_loader: Training data loader
    - test_loader: Test data loader
    - num_batches: Number of batches to analyze (None for all)
    """
    # Initialize collectors
    train_stats = defaultdict(list)
    test_stats = defaultdict(list)

    def collect_batch_stats(batch, stats_dict):
        # Node features
        stats_dict["node_features"].extend(batch.x.cpu().numpy())

        # Edge attributes
        stats_dict["edge_attr"].extend(batch.edge_attr.cpu().numpy())

        # Game state features
        stats_dict["down"].extend(batch.down.cpu().numpy())
        stats_dict["distance"].extend(batch.distance.cpu().numpy())
        stats_dict["quarter"].extend(batch.quarter.cpu().numpy())
        stats_dict["offense_team"].extend(batch.offense_team.cpu().numpy())

        # Graph structure features
        unique_batches = torch.unique(batch.batch)
        nodes_per_graph = [torch.sum(batch.batch == i).item() for i in unique_batches]
        edges_per_graph = [
            len(batch.edge_index[0][batch.batch[batch.edge_index[0]] == i])
            for i in unique_batches
        ]

        stats_dict["nodes_per_graph"].extend(nodes_per_graph)
        stats_dict["edges_per_graph"].extend(edges_per_graph)

        # Target distribution (only for eligible players)
        stats_dict["targets"].extend(batch.route_targets.cpu().numpy())

    # Collect statistics
    print("Collecting training set statistics...")
    for i, batch in enumerate(train_loader):
        if num_batches and i >= num_batches:
            break
        collect_batch_stats(batch, train_stats)

    print("Collecting test set statistics...")
    for i, batch in enumerate(test_loader):
        if num_batches and i >= num_batches:
            break
        collect_batch_stats(batch, test_stats)

    # Convert to numpy arrays
    for key in train_stats:
        train_stats[key] = np.array(train_stats[key])
        test_stats[key] = np.array(test_stats[key])

    # Analyze drift
    def plot_distribution_comparison(train_data, test_data, feature_name, bins=50):
        plt.figure(figsize=(10, 6))

        # Calculate histogram parameters
        min_val = min(train_data.min(), test_data.min())
        max_val = max(train_data.max(), test_data.max())

        plt.hist(
            train_data,
            bins=bins,
            alpha=0.5,
            label="Train",
            density=True,
            range=(min_val, max_val),
        )
        plt.hist(
            test_data,
            bins=bins,
            alpha=0.5,
            label="Test",
            density=True,
            range=(min_val, max_val),
        )

        plt.title(f"Distribution Comparison: {feature_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        # Perform KS test
        ks_statistic, p_value = ks_2samp(train_data, test_data)
        plt.text(
            0.05,
            0.95,
            f"KS statistic: {ks_statistic:.3f}\np-value: {p_value:.3e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.show()

    # Plot distributions for various features
    print("\nAnalyzing distributional drift...")

    # Node features
    for i in range(train_stats["node_features"][0].shape[0]):
        plot_distribution_comparison(
            train_stats["node_features"][:, i],
            test_stats["node_features"][:, i],
            f"Node Feature {i}",
        )

    # Edge attributes
    for i in range(train_stats["edge_attr"][0].shape[0]):
        plot_distribution_comparison(
            train_stats["edge_attr"][:, i],
            test_stats["edge_attr"][:, i],
            f"Edge Attribute {i}",
        )

    # Game state features
    for feature in ["down", "distance", "quarter"]:
        plot_distribution_comparison(
            train_stats[feature], test_stats[feature], feature.capitalize()
        )

    # Graph structure features
    for feature in ["nodes_per_graph", "edges_per_graph"]:
        plot_distribution_comparison(
            train_stats[feature], test_stats[feature], feature.replace("_", " ").title()
        )

    # Target distribution
    plot_distribution_comparison(
        train_stats["targets"], test_stats["targets"], "Route Targets"
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    for key in train_stats:
        print(f"\n{key.replace('_', ' ').title()}:")
        print(
            f"Train - Mean: {train_stats[key].mean():.3f}, Std: {train_stats[key].std():.3f}"
        )
        print(
            f"Test  - Mean: {test_stats[key].mean():.3f}, Std: {test_stats[key].std():.3f}"
        )


# Usage:
# analyze_distributional_drift(train_loader, test_loader)

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import entropy


def plot_metrics_across_datasets(
    evaluation_results: List[Tuple[Dict, Tuple]],
    dataset_names: List[str],
    metrics_to_plot: List[str] = ["accuracy", "weighted avg_f1-score"],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plots specified metrics across multiple datasets.

    Parameters:
    - evaluation_results: List of (metrics, data_tuple) from evaluate_route_predictions
    - dataset_names: Names/identifiers for each dataset
    - metrics_to_plot: List of metrics to visualize
    - figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)

    metrics_data = defaultdict(list)

    for metrics, _ in evaluation_results:
        for metric in metrics_to_plot:
            if "_" in metric:  # Handle nested metrics like 'weighted avg_f1-score'
                category, submetric = metric.split("_")
                value = metrics[category][submetric]
            else:
                value = metrics[metric]
            metrics_data[metric].append(value)

    x = np.arange(len(dataset_names))
    width = 0.8 / len(metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, metrics_data[metric], width, label=metric)

    plt.xlabel("Datasets")
    plt.ylabel("Score")
    plt.title("Metrics Comparison Across Datasets")
    plt.xticks(x + width * (len(metrics_to_plot) - 1) / 2, dataset_names)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_performance(
    metrics: Dict,
    data_tuple: Tuple,
    window_size: int = 100,
    stride: int = None,
    metrics_to_plot: List[str] = ["accuracy"],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plots rolling window performance metrics over an ordered dataset.

    Parameters:
    - metrics: Metrics dictionary from evaluate_route_predictions
    - data_tuple: (y_test, y_pred, y_pred_proba) tuple from evaluate_route_predictions
    - window_size: Size of the rolling window
    - stride: Number of samples to move the window forward (defaults to window_size//4)
    - metrics_to_plot: List of metrics to calculate and plot
    - figsize: Figure size for the plot
    """
    y_test, y_pred, y_pred_proba = data_tuple

    if stride is None:
        stride = max(window_size // 4, 1)  # Default stride is 1/4 of window size

    def calculate_window_metrics(start_idx: int, end_idx: int) -> Dict[str, float]:
        """Calculate metrics for the current window"""
        result = {}
        if "accuracy" in metrics_to_plot:
            result["accuracy"] = (
                y_pred[start_idx:end_idx] == y_test[start_idx:end_idx]
            ).mean()

        if "entropy" in metrics_to_plot:
            result["entropy"] = np.mean(
                [entropy(probs) for probs in y_pred_proba[start_idx:end_idx]]
            )

        if "top_2_accuracy" in metrics_to_plot:
            top_2_correct = 0
            for i in range(start_idx, end_idx):
                top_2 = np.argsort(y_pred_proba[i])[-2:]
                if y_test[i] in top_2:
                    top_2_correct += 1
            result["top_2_accuracy"] = top_2_correct / (end_idx - start_idx)

        if "log_likelihood" in metrics_to_plot:
            true_probs = np.array(
                [
                    probs[true]
                    for probs, true in zip(
                        y_pred_proba[start_idx:end_idx], y_test[start_idx:end_idx]
                    )
                ]
            )
            result["log_likelihood"] = np.mean(np.log(true_probs + 1e-10))

        return result

    # Calculate rolling metrics
    n_samples = len(y_test)
    windows = []
    rolling_metrics = defaultdict(list)

    current_start = 0
    while current_start + window_size <= n_samples:
        windows.append(
            current_start + window_size // 2
        )  # Use middle of window for x-axis
        metrics = calculate_window_metrics(current_start, current_start + window_size)
        for metric, value in metrics.items():
            rolling_metrics[metric].append(value)
        current_start += stride

    # Plotting
    plt.figure(figsize=figsize)

    for metric in metrics_to_plot:
        plt.plot(
            windows,
            rolling_metrics[metric],
            label=f"{metric} (window={window_size})",
            marker="o",
            markersize=3,
        )

    plt.xlabel("Sample Position")
    plt.ylabel("Score")
    plt.title("Rolling Performance Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Usage example:
"""
# For multiple datasets:
dataset_results = []
dataset_names = ['Train', 'Val', 'Test']

for dataset_loader in [train_loader, val_loader, test_loader]:
    metrics, data = evaluate_route_predictions(
        model=model,
        dataloader=dataset_loader,
        dataset=dataset,
        device='cuda'
    )
    dataset_results.append((metrics, data))

plot_metrics_across_datasets(
    evaluation_results=dataset_results,
    dataset_names=dataset_names,
    metrics_to_plot=['accuracy', 'weighted avg_f1-score']
)

# For rolling performance on single dataset:
metrics, data = evaluate_route_predictions(
    model=model,
    dataloader=test_loader,
    dataset=test_dataset,
    device='cuda'
)

plot_rolling_performance(
    metrics=metrics,
    data_tuple=data,
    window_size=100,
    stride=25,  # Move window forward by 25 samples each time
    metrics_to_plot=['accuracy', 'entropy', 'top_2_accuracy', 'log_likelihood']
)
"""


def evaluate_xgb_from_dataset(
    xgb_model, df, dataloader_, scaler, feature_encoders, device="cpu", debug=False
):

    all_predictions = []
    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader_:
            try:
                batch = batch.to(device)
                ids = list(
                    zip(
                        batch.game_id[batch.eligible_mask].cpu().tolist(),
                        batch.play_id[batch.eligible_mask].cpu().tolist(),
                        batch.player_ids[batch.eligible_mask].cpu().tolist(),
                    )
                )
                batch_df = filter_by_game_play_ids(df, ids)

                X, y, _, __ = prepare_route_prediction_data(
                    batch_df,
                    training=False,
                    feature_encoders=feature_encoders,
                    scaler=scaler,
                )
                probabilities = xgb_model.predict_proba(X)

                targets = torch.tensor(y)

                if debug:
                    batch_targets = batch.route_targets[batch.eligible_mask]
                    print(batch_targets)
                    print(targets)
                    raise Exception("Debugging")

                if isinstance(probabilities, torch.Tensor):
                    predictions = probabilities.argmax(dim=1).cpu().numpy()
                    probabilities = probabilities.cpu().numpy()
                else:
                    predictions = (
                        torch.tensor(probabilities).argmax(dim=1).cpu().numpy()
                    )

                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                # print(predictions)
                # print(probabilities)
                raise e

    all_predictions = np.concatenate(all_predictions)
    all_probabilities = np.concatenate(all_probabilities)
    all_targets = np.concatenate(all_targets)
    report = classification_report(y_true=all_targets, y_pred=all_predictions)
    top1k = top_k_accuracy_score(
        y_true=all_targets, y_score=all_probabilities, k=1, labels=list(range(0, 13))
    )
    top2k = top_k_accuracy_score(
        y_true=all_targets, y_score=all_probabilities, k=2, labels=list(range(0, 13))
    )

    return report, top1k, top2k
