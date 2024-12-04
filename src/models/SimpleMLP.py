import torch
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, config, state_size, action_size):
        super().__init__()
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.traversability_cost_size = 1
        self.traversasbility_breakdown_size = 8
        self.wheel_rpm_size = 4
        self.input_size = (
            self.state_size
            + self.action_size
            + self.traversability_cost_size
            + self.traversasbility_breakdown_size
            + self.wheel_rpm_size
        )
        hidden_dim = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_size),
        )

    def forward(
        self,
        state,
        actions,
        traversability_cost,
        traversability_breakdown,
        wheel_rpm,
    ):
        B, S = state.shape
        B, T, _ = actions.shape
        predictions = torch.zeros(B, T, S).to(state.device)
        for t in range(T):
            input = (
                torch.cat(
                    [
                        state,
                        actions[:, t, :],
                        traversability_cost[:, t, :],
                        traversability_breakdown[:, t, :],
                        wheel_rpm[:, t, :],
                    ],
                    dim=-1,
                )
                .float()
                .to(state.device)
            )
            next_state = self.mlp(input)
            predictions[:, t, :] = next_state
            state = next_state

        return predictions
