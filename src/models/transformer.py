import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class StateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StateEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return F.relu(self.fc(x))


class ActionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ActionEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return F.relu(self.fc(x))


class CostEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CostEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return F.relu(self.fc(x))


class BreakdownEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BreakdownEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return F.relu(self.fc(x))


class RPMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RPMEncoder, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return F.relu(self.fc(x))


class HeightmapEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(HeightmapEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)


class RGBMapEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(RGBMapEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)


class LatentTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers):
        super(LatentTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)


class AutoregressiveTransformerModel(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        cost_size,
        breakdown_size,
        rpm_size,
        H,
        W,
        hidden_size,
        nhead,
        num_layers,
        T,
    ):
        super(AutoregressiveTransformerModel, self).__init__()

        self.state_encoder = StateEncoder(state_size, hidden_size)
        self.action_encoder = ActionEncoder(action_size, hidden_size)
        self.cost_encoder = CostEncoder(cost_size, hidden_size)
        self.breakdown_encoder = BreakdownEncoder(breakdown_size, hidden_size)
        self.rpm_encoder = RPMEncoder(rpm_size, hidden_size)
        self.heightmap_encoder = HeightmapEncoder(pretrained=True, H=H, W=W)
        self.rgbmap_encoder = RGBMapEncoder(pretrained=True, H=H, W=W)

        latent_size = (
            hidden_size * 8
        )  # Case of concatenation of all latent sizes.
        self.latent_transformer = LatentTransformer(
            latent_size, hidden_size, nhead, num_layers
        )
        self.fc_out = nn.Linear(hidden_size, state_size)
        self.T = T

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
        B, T, _ = actions.shape

        # Encode all inputs
        encoded_state = self.state_encoder(state)
        encoded_action = self.action_encoder(actions.view(B * T, -1)).view(
            B, T, -1
        )
        encoded_cost = self.cost_encoder(
            traversability_cost.view(B * T, -1)
        ).view(B, T, -1)
        encoded_breakdown = self.breakdown_encoder(
            traversability_breakdown.view(B * T, -1)
        ).view(B, T, -1)
        encoded_rpm = self.rpm_encoder(wheel_rpm.view(B * T, -1)).view(B, T, -1)

        # Adjust heightmap and rgbmap to fit the encoder
        heightmap = heightmap.permute(0, 3, 1, 2)
        rgbmap = rgbmap.permute(0, 3, 1, 2)
        encoded_heightmap = self.heightmap_encoder(heightmap).view(B, -1)
        encoded_rgbmap = self.rgbmap_encoder(rgbmap).view(B, -1)

        # Concatenate all latent encodings
        latent_concat = torch.cat(
            (
                encoded_state.unsqueeze(1).repeat(1, T, 1),
                encoded_action,
                encoded_cost,
                encoded_breakdown,
                encoded_rpm,
                encoded_heightmap.unsqueeze(1).repeat(1, T, 1),
                encoded_rgbmap.unsqueeze(1).repeat(1, T, 1),
            ),
            dim=-1,
        )  # Shape (B, T, latent_size)

        latent_seq = latent_concat.permute(1, 0, 2)  # Shape (T, B, latent_size)
        tgt = torch.zeros_like(
            latent_seq[:-1]
        )  # Shifted target sequence for autoregression

        # Run the transformer for sequence prediction
        transformer_out = self.latent_transformer(latent_seq[:-1], tgt)
        prediction = self.fc_out(transformer_out)  # Shape (T, B, state_size)

        return prediction.permute(1, 0, 2)  # Shape (B, T, state_size)
