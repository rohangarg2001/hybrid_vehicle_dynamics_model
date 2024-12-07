import torch
from torch import nn
from src.models.transformer import HeightmapEncoder
from src.models.transformer import RGBMapEncoder


class SimpleMLP(nn.Module):
    def __init__(
        self,
        config,
        state_size,
        action_size,
        traversability_cost_size=1,
        traversasbility_breakdown_size=8,
        wheel_rpm_size=4,
        H=240,
        W=600,
    ):
        super().__init__()
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.traversability_cost_size = traversability_cost_size
        self.traversasbility_breakdown_size = traversasbility_breakdown_size
        self.wheel_rpm_size = wheel_rpm_size
        self.hidden_size = config["model"]["hidden_size"]
        if "height_map_12x12" in self.config["data"]["modalities"].keys():
            self.heightmap_encoder = HeightmapEncoder(
                self.hidden_size, pretrained=self.pretrained
            )
        if "image_left_color" in self.config["data"]["modalities"].keys():
            self.rgbmap_encoder = RGBMapEncoder(
                self.hidden_size, pretrained=self.pretrained
            )

        self.input_size = self.state_size + self.action_size
        if "traversability_cost" in self.config["data"]["modalities"].keys():
            self.input_size += self.traversability_cost_size
        if "traversability_breakdown" in self.config["data"]["modalities"].keys():
            self.input_size += self.traversasbility_breakdown_size
        if "wheel_rpm" in self.config["data"]["modalities"].keys():
            self.input_size += self.wheel_rpm_size
        if "height_map_12x12" in self.config["data"]["modalities"].keys():
            self.input_size += self.hidden_size
        if "image_left_color" in self.config["data"]["modalities"].keys():
            self.input_size += self.hidden_size

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
        heightmap,
        rgbmap,
    ):
        B, S = state.shape
        B, T, _ = actions.shape
        predictions = torch.zeros(B, T, S).to(state.device)
        for t in range(T):
            inp_tensors = [
                state,
                actions[:, t, :],
            ]
            if traversability_cost is not None:
                inp_tensors.append(traversability_cost)
            if traversability_breakdown is not None:
                inp_tensors.append(traversability_breakdown)
            if wheel_rpm is not None:
                inp_tensors.append(wheel_rpm)
            if heightmap is not None:
                inp_tensors.append(self.heightmap_encoder(heightmap))
            if rgbmap is not None:
                inp_tensors.append(self.rgbmap_encoder(rgbmap))
            input = (
                torch.cat(
                    inp_tensors,
                    dim=-1,
                )
                .float()
                .to(state.device)
            )
            next_state = self.mlp(input)
            predictions[:, t, :] = next_state
            state = next_state

        return predictions
