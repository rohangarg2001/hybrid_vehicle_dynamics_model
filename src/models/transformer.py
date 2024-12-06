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
    def __init__(self, hidden_size, pretrained=True):
        super(HeightmapEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x[:, :3, :, :]  # Simply taking the first 3 channels here
        x = self.feature_extractor(x)
        return self.fc(x.view(x.size(0), -1))


class RGBMapEncoder(nn.Module):
    def __init__(self, hidden_size, pretrained=True):
        super(RGBMapEncoder, self).__init__()
        resnet = resnet18(pretrained=pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)

        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.fc(x.view(x.size(0), -1))


class LatentTransformerEncoder(nn.Module):
    def __init__(self, input_size, nhead, num_layers):
        super(LatentTransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers,
        )

    def forward(self, src):
        return self.encoder(src)


class LatentTransformerDecoder(nn.Module):
    def __init__(self, input_size, nhead, num_layers):
        super(LatentTransformerDecoder, self).__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_size, nhead=nhead),
            num_layers=num_layers,
        )

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class AutoregressiveTransformerModel(nn.Module):
    def __init__(
        self,
        config,
        state_size,
        action_size,
        cost_size,
        breakdown_size,
        rpm_size,
        H=240,
        W=600,
    ):
        super(AutoregressiveTransformerModel, self).__init__()
        self.hidden_size = config["model"]["hidden_size"]
        self.nhead = config["model"]["nhead"]
        self.num_layers = config["model"]["num_layers"]
        self.pretrained = config["model"]["pretrained"]
        self.use_decoder = config["model"]["use_decoder"]

        self.state_encoder = StateEncoder(state_size, self.hidden_size)
        self.action_encoder = ActionEncoder(action_size, self.hidden_size)
        self.cost_encoder = CostEncoder(cost_size, self.hidden_size)
        self.breakdown_encoder = BreakdownEncoder(breakdown_size, self.hidden_size)
        self.rpm_encoder = RPMEncoder(rpm_size, self.hidden_size)
        self.heightmap_encoder = HeightmapEncoder(self.hidden_size, pretrained=self.pretrained)
        self.rgbmap_encoder = RGBMapEncoder(self.hidden_size, pretrained=self.pretrained)

        self.latent_size = self.hidden_size * 7  # Adjust according to your design
        self.latent_transformer_encoder = LatentTransformerEncoder(self.latent_size, self.nhead, self.num_layers)
        if self.use_decoder:
            self.latent_transformer_decoder = LatentTransformerDecoder(self.latent_size, self.nhead, self.num_layers)
        self.fc_out = nn.Linear(self.latent_size, state_size)

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
        encoded_state = self.state_encoder(state)  # Shape (B, hidden_size)
        encoded_action = self.action_encoder(actions.view(B * T, -1)).view(
            B, T, -1
        )  # Shape (B, T, hidden_size)
        encoded_cost = (
            self.cost_encoder(traversability_cost.view(B, -1))
            .unsqueeze(1)
            .expand(B, T, -1)
        )  # Shape (B, T, hidden_size)
        encoded_breakdown = (
            self.breakdown_encoder(traversability_breakdown.view(B, -1))
            .unsqueeze(1)
            .expand(B, T, -1)
        )  # Shape (B, T, hidden_size)
        encoded_rpm = (
            self.rpm_encoder(wheel_rpm.view(B, -1)).unsqueeze(1).expand(B, T, -1)
        )  # Shape (B, T, hidden_size)

        # Process heightmap and rgbmap
        heightmap = heightmap.permute(0, 3, 1, 2)  # (B, 4, H, W)
        rgbmap = rgbmap.permute(0, 3, 1, 2)  # (B, 3, H, W)

        # Encode heightmap and rgbmap
        encoded_heightmap = (
            self.heightmap_encoder(heightmap).unsqueeze(1).expand(B, T, -1)
        )  # Shape (B, T, 512)
        encoded_rgbmap = (
            self.rgbmap_encoder(rgbmap).unsqueeze(1).expand(B, T, -1)
        )  # Shape (B, T, 512)

        # Concatenate all latent encodings
        latent_concat = torch.cat(
            (
                encoded_state.unsqueeze(1).expand(
                    B, T, -1
                ),  # Shape (B, T, hidden_size)
                encoded_action,
                encoded_cost,
                encoded_breakdown,
                encoded_rpm,
                encoded_heightmap,
                encoded_rgbmap,
            ),
            dim=-1,
        )  # Shape (B, T, latent_size)

        latent_seq = latent_concat.permute(1, 0, 2)  # Shape (T, B, latent_size)

        # Run the Transformer encoder for sequence prediction
        transformer_out = self.latent_transformer_encoder(
            latent_seq
        )  # Output shape (T, B, hidden_size)

        if self.use_decoder:
            transformer_out = self.latent_transformer_decoder(
                transformer_out, transformer_out
            )

        # Use the entire sequence output of the transformer
        # Reshape transformer output to (B, T, latent_size)
        transformer_out = transformer_out.permute(1, 0, 2) 

        # Apply fc_out to each time step for the final prediction (B, T, state_size)
        prediction = self.fc_out(transformer_out)

        return prediction  # Shape (B, T, state_size)
