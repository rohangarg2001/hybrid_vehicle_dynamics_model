from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, config, state_size, action_size, horizon_size):
        super().__init__()
        self.config = config
        self.state_size = state_size
        self.horizon_size = horizon_size
        self.action_size = action_size
        self.input_size = state_size + action_size
        print(f"{self.input_size=}")
        hidden_dim = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_size),
        )

    def forward(self, x):
        return self.mlp(x)
