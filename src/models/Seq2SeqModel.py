import torch
import torch.nn as nn


class Seq2SeqModel(nn.Module):
    def __init__(self, config, input_dim, action_dim):
        super(Seq2SeqModel, self).__init__()
        self.config = config
        hidden_dim = config["model"]["hidden_size"]
        self.state_encoder = nn.Linear(input_dim, hidden_dim)
        self.full_decoder = nn.LSTM(
            action_dim + hidden_dim, hidden_dim, self.config["model"]["num_layers"]
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, initial_state, action_sequence):
        encoded_state = self.state_encoder(initial_state)
        batch_size, seq_len, action_dim = action_sequence.shape
        decoder_input = (
            torch.cat(
                [action_sequence, encoded_state.unsqueeze(1).repeat(1, seq_len, 1)],
                dim=2,
            )
            .float()
            .to(initial_state.device)
        )
        decoder_output, _ = self.full_decoder(decoder_input)
        predictions = self.output_layer(decoder_output)
        return predictions
