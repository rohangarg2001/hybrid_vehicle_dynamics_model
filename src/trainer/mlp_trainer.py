import pytorch_lightning as L
import torch
import torchmetrics as tm
from torch import nn

from src.models.SimpleMLP import SimpleMLP


class SimpleMLPModule(L.LightningModule):
    def __init__(self, config, state_size, action_size, horizon_size):
        super().__init__()
        self.config = config
        self.mlp = SimpleMLP(config, state_size, action_size, horizon_size)
        self.accuracy = tm.MeanSquaredError().to(self.device)

    def forward(self, state, actions):
        """
        Args:
            state (torch.Tensor): The input state at time t, shape (B, 13).
            actions (torch.Tensor): The actions input from time t to t+T, shape (B, T, 2).

        Returns:
            torch.Tensor: The predicted states from t+1 to t+T+1, shape (B, T, 13).
        """
        B, T, _ = actions.shape

        predictions = torch.zeros(B, T, 13).to(state.device)

        for t in range(T):
            input = (
                torch.cat([state, actions[:, t, :]], dim=-1).float().to(state.device)
            )
            next_state = self.mlp(input)
            predictions[:, t, :] = next_state
            state = next_state

        return predictions

    def train_loss(self, predictions, targets):
        if self.config["train"]["loss"] == "mse":
            return nn.MSELoss()(predictions, targets)
        raise ValueError(f"Invalid loss function: {self.config['train']['loss']}")

    def train_acc(self, predictions, targets):
        return self.accuracy(predictions, targets)

    def training_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["super_odom"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        predictions = self.forward(state, actions)
        loss = self.train_loss(predictions, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_mse",
            self.train_acc(predictions, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["super_odom"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        predictions = self.forward(state, actions)
        loss = self.train_loss(predictions, targets)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val_mse",
            self.train_acc(predictions, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["super_odom"].float()
        actions = batch["action_horizon"].float()
        targets = batch["ground_truth"].float()
        predictions = self.forward(state, actions)
        loss = self.train_loss(predictions, targets)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "test_mse",
            self.train_acc(predictions, targets),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        if self.config["train"]["optimizer"] == "adam":
            lr = self.config["train"]["learning_rate"]
            return torch.optim.Adam(self.parameters(), lr=lr)
