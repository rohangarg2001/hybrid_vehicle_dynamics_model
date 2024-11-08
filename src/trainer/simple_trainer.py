import pytorch_lightning as L
import torch
import torchmetrics as tm
from torch import nn

from src.models.Seq2SeqModel import Seq2SeqModel
from src.models.SimpleMLP import SimpleMLP


class TrainerModule(L.LightningModule):
    def __init__(self, config, state_size, action_size):
        super().__init__()
        self.config = config

        self.e2e_model = None
        if self.config["model"]["type"] == "simple_mlp":
            self.e2e_model = SimpleMLP(config, state_size, action_size)
        elif self.config["model"]["type"] == "seq2seq":
            self.e2e_model = Seq2SeqModel(config, state_size, action_size)
        self.accuracy = tm.MeanSquaredError().to(self.device)

    def forward(self, state, actions):
        """
        Args:
            state (torch.Tensor): The input state at time t, shape (B, state_size).
            actions (torch.Tensor): The actions input from time t to t+T, shape (B, T, action_size).

        Returns:
            torch.Tensor: The predicted states from t+1 to t+T+1, shape (B, T, state_size).
        """
        return self.e2e_model(state, actions)

    def train_loss(self, predictions, targets):
        if self.config["train"]["loss"] == "mse":
            return nn.MSELoss()(predictions, targets)
        raise ValueError(f"Invalid loss function: {self.config['train']['loss']}")

    def train_acc(self, predictions, targets):
        return self.accuracy(predictions, targets)

    def training_step(self, batch, batch_idx):
        # TODO: rename these to state, actions, targets
        state = batch["state"].float()
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
        state = batch["state"].float()
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
        state = batch["state"].float()
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
