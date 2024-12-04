import pytorch_lightning as L
import torch
import torchmetrics as tm
from torch import nn

from src.models.Seq2SeqModel import Seq2SeqModel
from src.models.SimpleMLP import SimpleMLP


class TrainerModule(L.LightningModule):
    def __init__(
        self,
        config,
        state_size,
        action_size,
        cost_size,
        breakdown_size,
        rpm_size,
        H,
        W,
    ):
        super().__init__()
        self.config = config

        self.e2e_model = None
        if self.config["model"]["type"] == "simple_mlp":
            self.e2e_model = SimpleMLP(
                config,
                state_size,
                action_size,
                cost_size,
                breakdown_size,
                rpm_size,
                H,
                W,
            )
        elif self.config["model"]["type"] == "seq2seq":
            self.e2e_model = Seq2SeqModel(
                config,
                state_size,
                action_size,
                cost_size,
                breakdown_size,
                rpm_size,
                H,
                W,
            )
        else:
            raise ValueError(
                f"Invalid model type: {self.config['model']['type']}"
            )
        self.mse = tm.MeanSquaredError()
        self.mae = tm.MeanAbsoluteError()
        self.mape = tm.MeanAbsolutePercentageError()

    def forward(
        self,
        state,
        actions,
        traversability_cost,
        traversability_breakdown,
        wheel_rpm,
        heightmap,
        rgbmap,
    ):
        """
        Args:
            state (torch.Tensor): The input state at time t, shape (B, state_size).
            actions (torch.Tensor): The actions input from time t to t+T, shape (B, T, action_size).
            traversability_cost (torch.Tensor): The traversability cost input from time t to t+T, shape (B, T, 1).
            traversability_breakdown (torch.Tensor): The traversability breakdown input from time t to t+T, shape (B, T, 8).
            wheel_rpm (torch.Tensor): The wheel rpm input from time t to t+T, shape (B, T, 4).
            heightmap (torch.Tensor): The heightmap input from time t to t+T, shape (B, H, W, 1).
            rgbmap (torch.Tensor): The rgbmap input from time t to t+T, shape (B, H, W, 3).
        Returns:
            torch.Tensor: The predicted states from t+1 to t+T+1, shape (B, T, state_size).
        """
        return self.e2e_model(
            state,
            actions,
            traversability_cost,
            traversability_breakdown,
            wheel_rpm,
            heightmap,
            rgbmap,
        )

    def train_loss(self, predictions, targets):
        _, T, _ = predictions.shape
        if self.config["train"]["loss"] == "mse":
            return nn.MSELoss()(predictions, targets) / T
        raise ValueError(
            f"Invalid loss function: {self.config['train']['loss']}"
        )

    def compute_metrics(self, predictions, targets, prefix: str = ""):
        _, T, _ = predictions.shape
        mse = self.mse(predictions, targets) / T
        mae = self.mae(predictions, targets) / T
        mape = self.mape(predictions, targets) / T
        rmse = torch.sqrt(mse)
        # r2 = self.r2(predictions, targets)
        return {
            f"{prefix}_normalized_mse": mse,
            f"{prefix}_normalized_mae": mae,
            f"{prefix}_normalized_mape": mape,
            f"{prefix}_normalized_rmse": rmse,
        }

    def training_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["state"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        traversability_cost = batch["traversability_cost"].float()
        wheel_rpm = batch["wheel_rpm"].float()
        traversability_breakdown = batch["traversability_breakdown"].float()
        height_map = batch["height_map"].float()
        rgb_map = batch["rgb_map"].float()

        predictions = self.forward(
            state,
            actions,
            traversability_cost,
            traversability_breakdown,
            wheel_rpm,
            height_map,
            rgb_map,
        )
        loss = self.train_loss(predictions, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        train_metrics = self.compute_metrics(
            predictions, targets, prefix="train"
        )
        for metric_name, metric_val in train_metrics.items():
            self.log(
                metric_name,
                metric_val,
                on_step=True,
                on_epoch=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["state"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        traversability_cost = batch["traversability_cost"].float()
        wheel_rpm = batch["wheel_rpm"].float()
        traversability_breakdown = batch["traversability_breakdown"].float()
        height_map = batch["height_map"].float()
        rgb_map = batch["rgb_map"].float()

        predictions = self.forward(
            state,
            actions,
            traversability_cost,
            traversability_breakdown,
            wheel_rpm,
            height_map,
            rgb_map,
        )
        loss = self.train_loss(predictions, targets)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        val_metrics = self.compute_metrics(predictions, targets, prefix="val")
        for metric_name, metric_val in val_metrics.items():
            self.log(
                metric_name,
                metric_val,
                on_step=True,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["state"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        traversability_cost = batch["traversability_cost"].float()
        wheel_rpm = batch["wheel_rpm"].float()
        traversability_breakdown = batch["traversability_breakdown"].float()
        height_map = batch["height_map"].float()
        rgb_map = batch["rgb_map"].float()

        predictions = self.forward(
            state,
            actions,
            traversability_cost,
            traversability_breakdown,
            wheel_rpm,
            height_map,
            rgb_map,
        )
        loss = self.train_loss(predictions, targets)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        test_metrics = self.compute_metrics(predictions, targets, prefix="test")
        for metric_name, metric_val in test_metrics.items():
            self.log(
                metric_name,
                metric_val,
                on_step=True,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.config["train"]["optimizer"] == "adam":
            lr = self.config["train"]["learning_rate"]
            return torch.optim.Adam(self.parameters(), lr=lr)
