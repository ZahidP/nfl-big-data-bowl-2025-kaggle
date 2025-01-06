import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, top_k_accuracy_score
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def accuracy_report_by_entropy(
    predictions_df, play_grouped, cutoff=2.1, k_values=[1, 2, 3]
):
    print(f"Initial DF size: {len(predictions_df)}")

    # Filter dataframes as before
    filtered_by_mean_play_entropy = predictions_df[
        predictions_df.play_id.isin(
            play_grouped["most_accurate_plays"].reset_index().play_id
        )
    ]
    filtered_by_play_and_row_entropy = filtered_by_mean_play_entropy[
        filtered_by_mean_play_entropy.entropy < cutoff
    ]
    filtered_by_row_entropy = predictions_df[predictions_df.entropy < cutoff]
    print(f"Final filtering size: {len(filtered_by_play_and_row_entropy)}")

    def calculate_topk_accuracy(actual, predicted_probs):
        predicted_probs = np.array(predicted_probs)

        topk_accuracies = {}
        for k in k_values:
            top_k_preds = np.argsort(-predicted_probs, axis=1)[:, :k]
            correct = [actual[i] in top_k_preds[i] for i in range(len(actual))]
            accuracy = np.mean(correct)
            topk_accuracies[f"Top-{k} Accuracy"] = f"{accuracy:.3f}"
        return topk_accuracies

    # Calculate and print Top-K accuracies for each filtered dataset
    print(
        f"\n=== Filtered by Row Entropy Only - Coverage: N = {len(filtered_by_row_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_row_entropy)/len(predictions_df), 2)} % ==="
    )
    print(
        classification_report(
            filtered_by_row_entropy.actual, filtered_by_row_entropy.predicted
        )
    )
    topk_metrics = calculate_topk_accuracy(
        filtered_by_row_entropy.actual.values,
        filtered_by_row_entropy.probabilities.tolist(),
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")

    print(
        f"\n=== Filtered by Mean Play Entropy - Coverage: N = {len(filtered_by_mean_play_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_mean_play_entropy)/len(predictions_df), 2)} % ==="
    )
    print(
        classification_report(
            filtered_by_mean_play_entropy.actual,
            filtered_by_mean_play_entropy.predicted,
        )
    )
    topk_metrics = calculate_topk_accuracy(
        filtered_by_mean_play_entropy.actual.values,
        filtered_by_mean_play_entropy.probabilities.tolist(),
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")

    print(
        f"\n=== Filtered by Both Play and Row Entropy - Coverage: N = {len(filtered_by_play_and_row_entropy)}/{len(predictions_df)} {round(100 * len(filtered_by_play_and_row_entropy)/len(predictions_df), 2)} % ===",
    )
    print(
        classification_report(
            filtered_by_play_and_row_entropy.actual,
            filtered_by_play_and_row_entropy.predicted,
        )
    )
    topk_metrics = calculate_topk_accuracy(
        filtered_by_play_and_row_entropy.actual.values,
        filtered_by_play_and_row_entropy.probabilities.tolist(),
    )
    print("Top-K Accuracy Metrics:")
    for metric, value in topk_metrics.items():
        print(f"{metric}: {value}")


def get_high_confidence_metrics(
    predictions_df, unique_routes, confidence_threshold=0.7, k=1
):
    """
    Calculate classification metrics for high-confidence predictions.

    Args:
        predictions_df: DataFrame with 'actual', 'predicted', and 'probabilities' columns
        confidence_threshold: Minimum probability threshold for high confidence predictions
        k: Number of top predictions to consider for top-k accuracy

    Returns:
        Dictionary containing classification report and top-k accuracy for high-confidence predictions
    """
    # Get max probability for each prediction
    predictions_df["max_probability"] = predictions_df["probabilities"].apply(max)

    # Filter for high confidence predictions
    high_conf_df = predictions_df[
        predictions_df["max_probability"] >= confidence_threshold
    ].copy()

    # Calculate metrics using all possible labels
    classification_metrics = classification_report(
        high_conf_df["actual"],
        high_conf_df["predicted"],
        labels=unique_routes,
        output_dict=True,
        zero_division=0,
    )

    # Calculate top-k accuracy for high confidence predictions
    def is_actual_in_topk(row, k):
        # Sort probabilities in descending order and get top k indices
        topk_indices = np.argsort(row["probabilities"])[-k:]
        # Convert actual to index (assuming actual is a label that maps to probability index)
        actual_idx = row["actual"]
        return actual_idx in topk_indices

    topk_accuracy = high_conf_df.apply(lambda x: is_actual_in_topk(x, k), axis=1).mean()

    # Prepare summary
    n_total = len(predictions_df)
    n_high_conf = len(high_conf_df)

    return {
        "classification_report": classification_metrics,
        f"top{k}_accuracy": topk_accuracy,
        "coverage": n_high_conf / n_total,
        "n_predictions": n_high_conf,
        "n_total": n_total,
        "confidence_threshold": confidence_threshold,
    }


def print_metrics_summary(metrics, k):
    """Pretty print the metrics summary"""
    accuracy_key = f"top{k}_accuracy"
    print(
        f"\nHigh Confidence Predictions (threshold >= {metrics['confidence_threshold']:.1%})"
    )
    print(
        f"Coverage: {metrics['coverage']:.1%} ({metrics['n_predictions']:,} / {metrics['n_total']:,} predictions)"
    )
    print(f"\nTop-{k} Accuracy: {metrics[accuracy_key]:.1%}")

    # Print classification report metrics
    report = metrics["classification_report"]
    print("\nClassification Report:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)

    # Print metrics for each class
    for class_name, class_metrics in report.items():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            print(
                f"{class_name:<15} {class_metrics['precision']:>10.3f} "
                f"{class_metrics['recall']:>10.3f} {class_metrics['f1-score']:>10.3f} "
                f"{class_metrics['support']:>10}"
            )

    # # Print averages
    # print("-" * 55)
    # print(report)
    # print(f"{'Accuracy':<15} {report['accuracy']:>10.3f}")
    # print(f"{'Macro Avg':<15} {report['macro avg']['precision']:>10.3f} "
    #       f"{report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")


def groupby_entropy_and_accuracy(predictions_df, k=100, extra_groupings=None):
    # Calculate accuracy for each play and sort by worst performing
    extra_groupings = [] if extra_groupings is None else extra_groupings
    predictions_df["is_correct"] = (
        predictions_df["actual"] == predictions_df["predicted"]
    )
    play_accuracy = predictions_df.groupby(
        ["play_id", "game_id"] + extra_groupings
    ).agg(
        {
            "is_correct": ["mean", "count"],  # get both accuracy and number of routes
            "actual_route": list,  # see what the actual routes were
            "predicted_route": list,  # see what we predicted
            "yardline": "first",  # include yardline for context
            "entropy": ["mean", "max"],
        }
    )

    # Flatten the column names
    play_accuracy.columns = [
        "accuracy",
        "num_routes",
        "actual_routes",
        "predicted_routes",
        "yardline",
        "entropy",
        "max_entropy",
    ]

    print(len(play_accuracy))

    # Sort by lowest accuracy (highest error rate)
    worst_plays = (
        play_accuracy.sort_values(
            "accuracy", ascending=True
        ).assign(  # ascending=True puts worst plays first
            error_rate=lambda x: 1 - x["accuracy"]
        )
    )[0:k]

    best_plays = (
        play_accuracy.sort_values(
            "accuracy", ascending=False
        ).assign(  # ascending=True puts worst plays first
            error_rate=lambda x: 1 - x["accuracy"]
        )
    )[0:k]

    uncertain_plays = (
        play_accuracy.sort_values(
            "entropy", ascending=False
        ).assign(  # ascending=True puts worst plays first
            error_rate=lambda x: x["entropy"]
        )
    )[0:k]

    certain_plays = (
        play_accuracy.sort_values(
            "entropy", ascending=True
        ).assign(  # ascending=True puts worst plays first
            error_rate=lambda x: x["accuracy"]
        )
    )[0:k]

    return {
        "least_accurate_plays": worst_plays,
        "most_accurate_plays": best_plays,
        "high_entropy_plays": uncertain_plays,
        "low_entropy_plays": certain_plays,
        "play_accuracy": play_accuracy,
    }


def plot_entropy_and_accuracy(
    least_accurate_plays,
    most_accurate_plays,
    high_entropy_plays,
    low_entropy_plays,
    title_update="Play",
    ax1=None,
    ax2=None,
    ax3=None,
    plot_type="kde",  # Add plot_type parameter
    *args,
    **kwargs,
):

    # Create a figure with 3 subplots arranged vertically
    if ax1 is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 2, figsize=(8, 10))

    # Helper function to choose between kde and histogram
    def plot_distribution(data, x, label, ax):
        if plot_type == "kde":
            sns.kdeplot(data=data, x=x, fill=True, alpha=0.5, label=label, ax=ax)
        else:  # histogram
            sns.histplot(data=data, x=x, alpha=0.5, label=label, ax=ax, stat="density")

    # First subplot - Entropy
    plot_distribution(
        least_accurate_plays, "entropy", f"Low Accuracy {title_update}s", ax1
    )
    plot_distribution(
        most_accurate_plays, "entropy", f"High Accuracy {title_update}s", ax1
    )
    ax1.set_title(f"Distribution of {title_update} Entropy", fontsize=12)
    ax1.set_xlabel("Entropy", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.legend(fontsize=9)

    # Second subplot - Max Entropy
    plot_distribution(
        least_accurate_plays, "max_entropy", f"Low Accuracy {title_update}s", ax2
    )
    plot_distribution(
        most_accurate_plays, "max_entropy", f"High Accuracy {title_update}s", ax2
    )
    ax2.set_title(f"Distribution of {title_update} Max Entropy", fontsize=12)
    ax2.set_xlabel("Max Entropy", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(fontsize=9)

    # Third subplot - Accuracy
    plot_distribution(
        low_entropy_plays, "accuracy", f"Low Entropy {title_update}s", ax3
    )
    plot_distribution(
        high_entropy_plays, "accuracy", f"High Entropy {title_update}s", ax3
    )
    ax3.set_title(f"Distribution of {title_update} Accuracy", fontsize=12)
    ax3.set_xlabel("Accuracy", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.legend(fontsize=9)

    # Adjust layout to prevent overlap
    if ax1 is None:
        plt.tight_layout()
        plt.show()


def get_high_confidence_metrics(
    predictions_df, unique_routes, confidence_threshold=0.7, k=1
):
    """
    Calculate classification metrics for high-confidence predictions.

    Args:
        predictions_df: DataFrame with 'actual', 'predicted', and 'probabilities' columns
        unique_routes: List of all possible route labels
        confidence_threshold: Minimum probability threshold for high confidence predictions
        k: Number of top predictions to consider for top-k accuracy

    Returns:
        Dictionary containing classification report and top-k accuracy for high-confidence predictions
    """
    # Get max probability for each prediction
    predictions_df["max_probability"] = predictions_df["probabilities"].apply(max)

    # Filter for high confidence predictions
    high_conf_df = predictions_df[
        predictions_df["max_probability"] >= confidence_threshold
    ].copy()

    # Calculate top-k accuracy for high confidence predictions
    def is_actual_in_topk(row, k):
        # Get indices of top k probabilities
        topk_indices = np.argsort(row["probabilities"])[-k:]
        # actual is already an index, use it directly
        return row["actual"] in topk_indices

    # Map actual and predicted to same label space
    def ensure_label_consistency(df):
        # Map actual and predicted values to indices in unique_routes if they aren't already
        if not isinstance(df["actual"].iloc[0], (int, np.integer)):
            df["actual_idx"] = df["actual"].apply(lambda x: unique_routes.index(x))
        else:
            df["actual_idx"] = df["actual"]

        if not isinstance(df["predicted"].iloc[0], (int, np.integer)):
            df["predicted_idx"] = df["predicted"].apply(
                lambda x: unique_routes.index(x)
            )
        else:
            df["predicted_idx"] = df["predicted"]
        return df

    high_conf_df = ensure_label_consistency(high_conf_df)

    # Map integer indices to route names for classification report
    high_conf_df["actual_route"] = high_conf_df["actual"].apply(
        lambda x: unique_routes[x]
    )
    high_conf_df["predicted_route"] = high_conf_df["predicted"].apply(
        lambda x: unique_routes[x]
    )

    # Calculate metrics using mapped route names
    classification_metrics = classification_report(
        high_conf_df["actual_route"],
        high_conf_df["predicted_route"],
        labels=unique_routes,
        output_dict=True,
        zero_division=0,
    )

    # Remove the label mapping code since we're using strings directly
    mapped_metrics = classification_metrics

    # Calculate top-k accuracies for k=1,2,3
    def calc_topk_accuracy(df, k):
        return df.apply(lambda x: is_actual_in_topk(x, k), axis=1).mean()

    topk_accuracies = {
        f"top{k}_accuracy": calc_topk_accuracy(high_conf_df, k) for k in [1, 2, 3]
    }

    for key, v in topk_accuracies.items():
        print(f"{key}: {round(v, 2)}")

    # Prepare summary
    n_total = len(predictions_df)
    n_high_conf = len(high_conf_df)

    # Add debug information
    debug_info = {
        "sample_actual": high_conf_df["actual"].head().tolist(),
        "sample_predicted": high_conf_df["predicted"].head().tolist(),
        "sample_probabilities_shape": [
            len(p) for p in high_conf_df["probabilities"].head()
        ],
        "unique_actual_values": high_conf_df["actual"].nunique(),
        "unique_predicted_values": high_conf_df["predicted"].nunique(),
        "number_of_routes": len(unique_routes),
    }

    return {
        "classification_report": mapped_metrics,
        f"top{k}_accuracy": topk_accuracies[f"top{k}_accuracy"],
        "coverage": n_high_conf / n_total,
        "n_predictions": n_high_conf,
        "n_total": n_total,
        "confidence_threshold": confidence_threshold,
        "debug_info": debug_info,
    }


def create_corr_plot(i, all_gnn_preds, all_xgb_preds, all_true_labels, route_names):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get top predictions for each model
    gnn_top = np.argmax(all_gnn_preds, axis=1)
    xgb_top = np.argmax(all_xgb_preds, axis=1)
    
    # True negatives plot
    true_negatives = all_true_labels != i
    neg_df = pd.DataFrame({
        'gnn': all_gnn_preds[true_negatives, i],
        'xgb': all_xgb_preds[true_negatives, i],
        'gnn_pred': gnn_top[true_negatives] == i,
        'xgb_pred': xgb_top[true_negatives] == i,
        'both_pred': (gnn_top[true_negatives] == i) & (xgb_top[true_negatives] == i)
    })
    
    # Color code based on predictions
    colors = np.where(neg_df['both_pred'], 'purple',
                     np.where(neg_df['gnn_pred'], 'blue',
                             np.where(neg_df['xgb_pred'], 'red', 'gray')))
    
    ax1.scatter(neg_df['gnn'], neg_df['xgb'], c=colors, alpha=0.5)
    ax1.set_title(f'True Negatives (Route {route_names[i]})')
    ax1.set_xlabel('GNN Probability')
    ax1.set_ylabel('XGBoost Probability')
    
    # True positives plot
    true_positives = all_true_labels == i
    pos_df = pd.DataFrame({
        'gnn': all_gnn_preds[true_positives, i],
        'xgb': all_xgb_preds[true_positives, i],
        'gnn_pred': gnn_top[true_positives] == i,
        'xgb_pred': xgb_top[true_positives] == i,
        'both_pred': (gnn_top[true_positives] == i) & (xgb_top[true_positives] == i)
    })
    
    colors = np.where(pos_df['both_pred'], 'purple',
                     np.where(pos_df['gnn_pred'], 'blue',
                             np.where(pos_df['xgb_pred'], 'red', 'gray')))
    
    ax2.scatter(pos_df['gnn'], pos_df['xgb'], c=colors, alpha=0.5)
    ax2.set_title(f'True Positives (Route {route_names[i]})')
    ax2.set_xlabel('GNN Probability')
    ax2.set_ylabel('XGBoost Probability')
    
    # Add diagonal line to both plots
    for ax in [ax1, ax2]:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Neither predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='GNN predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='XGB predicted', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', label='Both predicted', markersize=10)
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax2.legend(handles=legend_elements, loc='upper left')
    
    # Add correlation coefficient as text
    corr = pearsonr(all_gnn_preds[:, i], all_xgb_preds[:, i])[0]
    plt.suptitle(f'Model Comparison for Route {route_names[i]} (correlation: {corr:.3f})', y=1.05)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for Route {route_names[i]}:")
    print(f"Number of true positives: {true_positives.sum()}")
    print(f"Number of true negatives: {true_negatives.sum()}")
    print(f"Overall correlation: {corr:.3f}")
    
    # Prediction statistics for true positives
    if true_positives.sum() > 0:
        n_both = pos_df['both_pred'].sum()
        n_gnn = pos_df['gnn_pred'].sum() - n_both
        n_xgb = pos_df['xgb_pred'].sum() - n_both
        n_neither = len(pos_df) - n_gnn - n_xgb - n_both
        print("\nTrue Positive Predictions:")
        print(f"Both models correct: {n_both}")
        print(f"Only GNN correct: {n_gnn}")
        print(f"Only XGB correct: {n_xgb}")
        print(f"Neither correct: {n_neither}")
    
    # Prediction statistics for true negatives
    if true_negatives.sum() > 0:
        n_both = neg_df['both_pred'].sum()
        n_gnn = neg_df['gnn_pred'].sum() - n_both
        n_xgb = neg_df['xgb_pred'].sum() - n_both
        n_neither = len(neg_df) - n_gnn - n_xgb - n_both
        print("\nTrue Negative Predictions:")
        print(f"Both predicted incorrectly: {n_both}")
        print(f"Only GNN predicted incorrectly: {n_gnn}")
        print(f"Only XGB predicted incorrectly: {n_xgb}")
        print(f"Neither predicted incorrectly: {n_neither}")