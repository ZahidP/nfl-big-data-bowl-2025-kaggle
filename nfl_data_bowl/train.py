"""Training loops"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MultiRouteLoss(nn.Module):
    def __init__(self, num_route_classes, class_weights=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


def add_regularization(model, loss, l1_lambda=0.0, l2_lambda=0.0):
    """
    Add L1 and L2 regularization to the loss.

    Parameters:
    - model: The neural network model
    - loss: The current loss value
    - l1_lambda: L1 regularization strength
    - l2_lambda: L2 regularization strength

    Returns:
    - Total loss with regularization
    """
    if l1_lambda == 0.0 and l2_lambda == 0.0:
        return loss

    # Initialize regularization terms
    l1_reg = torch.tensor(0.0, device=loss.device)
    l2_reg = torch.tensor(0.0, device=loss.device)

    for param in model.parameters():
        if param.requires_grad:
            if l1_lambda > 0:
                l1_reg += torch.sum(torch.abs(param))
            if l2_lambda > 0:
                l2_reg += torch.sum(param**2)

    total_loss = loss + (l1_lambda * l1_reg) + (l2_lambda * l2_reg)
    return total_loss


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy


class MultiRouteLoss(nn.Module):
    def __init__(self, num_route_classes, class_weights=None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)


def add_regularization(model, loss, l1_lambda=0.0, l2_lambda=0.0):
    """
    Add L1 and L2 regularization to the loss.

    Parameters:
    - model: The neural network model
    - loss: The current loss value
    - l1_lambda: L1 regularization strength
    - l2_lambda: L2 regularization strength

    Returns:
    - Total loss with regularization
    """
    if l1_lambda == 0.0 and l2_lambda == 0.0:
        return loss

    # Initialize regularization terms
    l1_reg = torch.tensor(0.0, device=loss.device)
    l2_reg = torch.tensor(0.0, device=loss.device)

    for param in model.parameters():
        if param.requires_grad:
            if l1_lambda > 0:
                l1_reg += torch.sum(torch.abs(param))
            if l2_lambda > 0:
                l2_reg += torch.sum(param**2)

    total_loss = loss + (l1_lambda * l1_reg) + (l2_lambda * l2_reg)
    return total_loss


def evaluate_model(model, loader, criterion, device, l1_lambda=0.0, l2_lambda=0.0):
    """
    Evaluate the model on a given data loader.

    Parameters:
    - model: The neural network model
    - loader: DataLoader for evaluation
    - criterion: Loss criterion
    - device: Device to evaluate on
    - l1_lambda: L1 regularization strength
    - l2_lambda: L2 regularization strength

    Returns:
    - Average loss on the evaluation set
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            predictions = output["route_predictions"]
            targets = batch.route_targets[batch.eligible_mask]

            loss = criterion(predictions, targets)
            loss = add_regularization(model, loss, l1_lambda, l2_lambda)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    if num_batches > 0:
        return total_loss / num_batches
    else:
        return total_loss


def train_route_predictor(
    model,
    train_loader,
    holdout_loader,
    optimizer,
    num_classes,
    num_epochs,
    device="cpu",
    lr_patience=5,
    lr_factor=0.5,
    lr_min=1e-6,
    l1_lambda=0.0,
    l2_lambda=0.0,
    early_stopping_patience=10,
    debug=False,
    allow_overfit=False,
    class_weights=None,
):
    """
    Training loop for route prediction with learning rate decay, regularization,
    and early stopping based on holdout performance.

    Parameters:
    - model: The neural network model
    - train_loader: DataLoader for training data
    - holdout_loader: DataLoader for holdout validation data
    - optimizer: Optimizer instance
    - num_classes: Number of route classes
    - num_epochs: Maximum number of epochs
    - device: Device to train on
    - lr_patience: Number of epochs without improvement before reducing learning rate
    - lr_factor: Factor to multiply learning rate by when decaying
    - lr_min: Minimum learning rate
    - l1_lambda: L1 regularization strength
    - l2_lambda: L2 regularization strength
    - early_stopping_patience: Number of epochs without improvement before stopping
    - debug: Enable debug printing

    Returns:
    - Dictionary containing training history and best model state
    """
    model.train()
    criterion = MultiRouteLoss(num_classes, class_weights=class_weights)

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=lr_min,
        verbose=True,
    )

    print("Training model")
    best_holdout_loss = float("inf")
    epochs_without_improvement = 0
    training_history = []
    best_model_state = None
    impatient_stop = False

    loss = None

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        # print(f"Beginning epoch: {epoch + 1}")

        # Training phase
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)
            targets = batch.route_targets[batch.eligible_mask]

            targets = targets.long()

            if debug:
                print(f"predictions shape: {output['route_predictions'].shape}")
                print(output["route_predictions"])
                print(f"targets shape: {targets.shape}")
                print(f"targets: {targets}")
                print(f"predictions 0:5: {output['route_predictions'][0:5]}")
                print(f"targets 0:5: {targets[0:5]}")

            predictions = output["route_predictions"]
            try:
                if len(predictions):
                    # Calculate base loss
                    loss = criterion(predictions, targets)

                    # Add regularization if specified
                    loss = add_regularization(model, loss, l1_lambda, l2_lambda)

                    # Backpropagate
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                else:
                    # {str(batch.play_id)}
                    print(
                        f"No predictions: {len(predictions)} for {len(targets)} in batch {num_batches}"
                    )
            except Exception as e:
                print(num_batches)
                print(batch)
                raise e

            num_batches += 1

            if torch.isnan(loss):
                print("NaN loss detected - stopping training")
                print(f"Last loss value: {loss.item()}")
                print(f"Predictions: {predictions}")
                print(f"Targets: {targets}")
                print(f"Batch num: {num_batches}")
                # Restore best model if available
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                return {
                    "history": training_history,
                    "best_model_state": best_model_state,
                }

            if debug:
                print(f"Loss: {loss.item()}")
                print(f"Predictions: {predictions}")
                print(f"Targets: {targets}")
                if num_batches > 3:
                    raise Exception("Debug mode - stopping after 3 batches")

        avg_train_loss = epoch_loss / num_batches

        # Evaluate on holdout set
        holdout_loss = evaluate_model(
            model, holdout_loader, criterion, device, l1_lambda, l2_lambda
        )

        # Put model back in training mode after evaluation
        model.train()

        current_lr = optimizer.param_groups[0]["lr"]

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "holdout_loss": holdout_loss,
                "lr": current_lr,
            }
        )

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Holdout Loss: {holdout_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )

        # Update learning rate scheduler based on holdout loss
        if allow_overfit:
            scheduler.step(avg_train_loss)
        else:
            scheduler.step(holdout_loss)
        state_dict = model.state_dict()
        # Early stopping check on holdout performance
        if holdout_loss < best_holdout_loss:
            best_holdout_loss = holdout_loss
            epochs_without_improvement = 0
            # Save best model weights

            best_model_state = copy.deepcopy(state_dict)
        else:
            epochs_without_improvement += 1
        if not allow_overfit:
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                impatient_stop = True
                break

        # Check if learning rate has become too small
        if current_lr <= lr_min:
            print(
                f"Learning rate {current_lr} has reached minimum {lr_min}. Stopping training."
            )
            break

    # Restore best model weights
    if best_model_state is not None and impatient_stop:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(state_dict)

    return {"history": training_history, "best_model_state": best_model_state}
