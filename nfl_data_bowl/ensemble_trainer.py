from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from typing import Dict, Optional, Union, Literal
from sklearn.metrics import log_loss, accuracy_score

# from nfl_data_bowl.deprecated.train_xgb_gpt import prepare_route_prediction_data
from nfl_data_bowl.train_xgb import prepare_route_prediction_data
from nfl_data_bowl.data_utils.data_common import filter_by_game_play_ids


@dataclass
class EnsembleConfig:
    meta_type: Literal["nn", "gbm"] = "nn"
    # NN params
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5
    # GBM params
    gbm_params: Dict = None
    scheduler_type = "cosine"
    gamma = 0.7
    min_lr = 0.00001

    def __post_init__(self):
        if self.meta_type == "gbm" and not self.gbm_params:
            self.gbm_params = {
                "objective": "multi:softprob",
                "learning_rate": 0.1,
                "max_depth": 4,
                "n_estimators": 100,
                "early_stopping_rounds": 10,
            }


class MetaLearner(nn.Module):
    def __init__(self, num_classes: int, meta_type: str, gbm_params: Dict = None):
        super().__init__()
        self.meta_type = meta_type
        self.hidden_dim = 64
        if meta_type == "nn":
            self.model = nn.Sequential(
                nn.Linear(2 * num_classes, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.Linear(self.hidden_dim // 4, num_classes),
            )
        else:
            self.model = xgb.XGBClassifier(num_class=num_classes, **gbm_params)

    def forward(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.meta_type == "nn":
            return self.model(x)
        return self.model.predict_proba(x)


class RouteEnsemble(nn.Module):
    def __init__(
        self,
        xgb_model,
        gnn_model,
        config: EnsembleConfig,
        num_classes: int,
        feature_encoders,
        device: str = "cuda",
        scaler=None,
        debug=False,
    ):
        super().__init__()
        self.xgb_model = xgb_model
        self.gnn_model = gnn_model
        self.config = config
        self.device = device
        self.feature_encoders = feature_encoders
        self.scaler = scaler
        self.debug = debug
        self.meta_learner = MetaLearner(
            num_classes=num_classes,
            meta_type=config.meta_type,
            gbm_params=config.gbm_params,
        ).to(device)

        if config.meta_type == "nn":
            self.optimizer = torch.optim.AdamW(
                self.meta_learner.parameters(), lr=config.lr
            )
            self.criterion = nn.CrossEntropyLoss()
            # Initialize scheduler based on type
            if config.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.epochs, eta_min=config.min_lr
                )
            else:  # 'step'
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=config.step_size, gamma=config.gamma
                )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass compatible with evaluation function"""
        # Get base model predictions
        xgb_probs, gnn_probs, y, X_prepared = self._get_single_batch_base_predictions(
            batch, self.current_plays_df
        )

        # Combine predictions using meta-learner
        if self.config.meta_type == "nn":
            xgb_probs = torch.FloatTensor(xgb_probs)
            gnn_probs = torch.FloatTensor(gnn_probs)
            x = torch.cat([xgb_probs, gnn_probs], dim=1)
            combined_logits = self.meta_learner.model(x)  # Get logits before softmax
            return {
                "route_predictions": combined_logits,
                "gnn": gnn_probs,
                "xgb": xgb_probs,
                "target": y,
                "X_prepared": X_prepared,
            }
        else:
            x = np.concatenate([xgb_probs, gnn_probs], axis=1)
            combined_probs = self.meta_learner.model.predict_proba(x)
            # Convert to logits for compatibility
            combined_logits = torch.FloatTensor(np.log(combined_probs + 1e-10)).to(
                self.device
            )
            return {
                "route_predictions": combined_probs,
                "gnn": torch.FloatTensor(gnn_probs),
                "xgb": torch.FloatTensor(xgb_probs),
                "target": y,
                "X_prepared": X_prepared,
            }

    def _get_single_batch_base_predictions(self, batch, plays_df) -> tuple:
        """Get predictions from base models"""
        all_xgb, all_gnn, all_labels = [], [], []

        # Get corresponding plays for XGBoost
        ids = list(
            zip(
                batch.game_id[batch.eligible_mask].cpu().tolist(),
                batch.play_id[batch.eligible_mask].cpu().tolist(),
                batch.player_ids[batch.eligible_mask].cpu().tolist(),
            )
        )
        batch_df = filter_by_game_play_ids(plays_df, ids)

        # GNN predictions
        gnn_batch = batch.to(self.device)
        gnn_out = self.gnn_model(gnn_batch)["route_predictions"]
        gnn_probs = torch.softmax(gnn_out, dim=-1).detach().cpu().numpy()

        # XGB predictions
        X, y, _, _2 = prepare_route_prediction_data(
            batch_df,
            training=False,
            feature_encoders=self.feature_encoders,
            scaler=self.scaler,
        )
        self.X_prepared = X
        xgb_probs = self.xgb_model.predict_proba(X)

        try:
            batch_vals = batch.route_targets[batch.eligible_mask]
            y_vals = torch.tensor(y).to(self.device)
            torch.testing.assert_close(batch_vals, y_vals)
        except AssertionError as e:
            print(batch_vals)
            print(y_vals)
            raise e

        return xgb_probs, gnn_probs, y, self.X_prepared

    def _get_base_predictions(self, loader, plays_df) -> tuple:
        """Get predictions from base models"""
        all_xgb, all_gnn, all_labels = [], [], []

        for batch in loader:
            xgb_probs, gnn_probs, y, X_prepared = (
                self._get_single_batch_base_predictions(batch, plays_df)
            )
            all_xgb.append(xgb_probs)
            all_gnn.append(gnn_probs)
            all_labels.append(y)

        xgb_preds = np.concatenate(all_xgb)
        gnn_preds = np.concatenate(all_gnn)
        labels = np.concatenate(all_labels)

        if self.config.meta_type == "nn":
            xgb_preds = torch.FloatTensor(xgb_preds).to(self.device)
            gnn_preds = torch.FloatTensor(gnn_preds).to(self.device)
            labels = torch.LongTensor(labels).to(self.device)

        return xgb_preds, gnn_preds, labels

    def _train_nn_epoch(self, xgb_preds, gnn_preds, labels):
        indices = torch.randperm(len(labels))
        losses = []

        # Debug prints at start
        # print(f"XGB preds range: {xgb_preds.min().item():.3f} to {xgb_preds.max().item():.3f}")
        # print(f"GNN preds range: {gnn_preds.min().item():.3f} to {gnn_preds.max().item():.3f}")
        # print(f"Labels range: {labels.min().item():.3f} to {labels.max().item():.3f}")

        for i in range(0, len(labels), self.config.batch_size):
            idx = indices[i : i + self.config.batch_size]
            x = torch.cat([xgb_preds[idx], gnn_preds[idx]], dim=1)

            self.optimizer.zero_grad()
            out = self.meta_learner(x)

            # Debug prints
            # print(f"Output range: {out.min().item():.3f} to {out.max().item():.3f}")

            loss = self.criterion(out, labels[idx])
            loss.backward()

            # Check gradients
            total_grad = 0
            for param in self.meta_learner.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().mean().item()
            # print(f"Average gradient magnitude: {total_grad}")

            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    def _evaluate(self, xgb_preds, gnn_preds, labels):
        if self.config.meta_type == "nn":
            self.meta_learner.eval()
            with torch.no_grad():
                x = torch.cat([xgb_preds, gnn_preds], dim=1)
                out = self.meta_learner(x)
                loss = self.criterion(out, labels).item()
                preds = out.argmax(dim=1).cpu()
                acc = accuracy_score(labels.cpu(), preds)
        else:
            x = np.concatenate([xgb_preds, gnn_preds], axis=1)
            out = self.meta_learner.model.predict_proba(x)
            loss = log_loss(labels, out)
            acc = accuracy_score(labels, out.argmax(axis=1))

        return {"loss": loss, "accuracy": acc}

    def train_ensemble(self, train_loader, train_df, val_loader=None, val_df=None):
        """Train the ensemble"""
        print("Caching base model predictions...")
        xgb_preds, gnn_preds, labels = self._get_base_predictions(
            train_loader, train_df
        )
        val_preds = (
            None
            if val_loader is None
            else self._get_base_predictions(val_loader, val_df)
        )
        train_preds = xgb_preds, gnn_preds, labels

        print(f"Training {self.config.meta_type.upper()} meta-learner...")
        best_val_loss = float("inf")
        patience_counter = 0

        if self.config.meta_type == "nn":
            for epoch in range(self.config.epochs):
                self.meta_learner.train()
                train_loss = self._train_nn_epoch(xgb_preds, gnn_preds, labels)

                metrics = {"train_loss": train_loss}
                if val_preds:
                    val_metrics = self._evaluate(*val_preds)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        patience_counter = 0
                        self.save_checkpoint("best_model.pt")
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.patience:
                        print("Early stopping triggered")
                        break

                print(
                    f"Epoch {epoch+1}/{self.config.epochs} -",
                    " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items()),
                )
        else:
            X_train = np.concatenate([train_preds[0], train_preds[1]], axis=1)
            eval_set = None
            if val_preds:
                X_val = np.concatenate([val_preds[0], val_preds[1]], axis=1)
                eval_set = [(X_train, train_preds[2]), (X_val, val_preds[2])]

            self.meta_learner.model.fit(
                X_train,
                train_preds[2],
                eval_set=eval_set,
                verbose=True,
            )
            # Step 5: Generate predictions and metrics
            self.test_preds = self.meta_learner.model.predict(X_val)
            self.test_true = val_preds[2]
            self.test_pred_a = self.meta_learner.model.predict_proba(X_val)
            self.train_preds = self.meta_learner.model.predict(X_train)
            self.train_pred_a = self.meta_learner.model.predict_proba(X_train)
            self.train_true = train_preds[2]

    def predict(self, loader, plays_df):
        """Generate ensemble predictions"""
        xgb_preds, gnn_preds, labels = self._get_base_predictions(loader, plays_df)

        if self.config.meta_type == "nn":
            self.meta_learner.eval()
            with torch.no_grad():
                x = torch.cat([xgb_preds, gnn_preds], dim=1)
                ensemble_preds = self.meta_learner(x).cpu().numpy()
        else:
            x = np.concatenate([xgb_preds, gnn_preds], axis=1)
            ensemble_preds = self.meta_learner.model.predict_proba(x)

        return {
            "ensemble_preds": ensemble_preds,
            "xgb_preds": (
                xgb_preds.cpu().numpy() if torch.is_tensor(xgb_preds) else xgb_preds
            ),
            "gnn_preds": (
                gnn_preds.cpu().numpy() if torch.is_tensor(gnn_preds) else gnn_preds
            ),
            "labels": labels.cpu().numpy() if torch.is_tensor(labels) else labels,
        }

    def set_plays_df(self, df):
        """Set the current plays DataFrame for forward pass"""
        self.current_plays_df = df

    def save_checkpoint(self, path: str):
        if self.config.meta_type == "nn":
            torch.save(
                {
                    "model_state": self.meta_learner.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "config": self.config,
                },
                path,
            )
        else:
            self.meta_learner.model.save_model(path)

    def load_checkpoint(self, path: str):
        if self.config.meta_type == "nn":
            checkpoint = torch.load(path)
            self.meta_learner.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            self.meta_learner.model.load_model(path)


# Example usage:
"""
config = EnsembleConfig(
    meta_type='nn',  # or 'gbm'
    lr=1e-3,
    epochs=10,
    patience=5
)

ensemble = RouteEnsemble(
    xgb_model=xgb_model,
    gnn_model=gnn_model,
    config=config,
    num_classes=5
)

# Train
ensemble.train(
    train_loader=train_loader,
    train_df=train_df,
    val_loader=val_loader,
    val_df=val_df
)

# Predict
results = ensemble.predict(test_loader, test_df)
"""
