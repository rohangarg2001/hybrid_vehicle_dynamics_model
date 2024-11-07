import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, control_dim, hidden_dim, output_dim, num_layers=1):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.LSTM(control_dim + hidden_dim, hidden_dim, num_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, initial_state, control_sequence):
        encoded_state = self.encoder(initial_state)        
        batch_size, seq_len, control_dim = control_sequence.shape
        decoder_input = torch.cat([control_sequence, encoded_state.unsqueeze(1).repeat(1, seq_len, 1)], dim=2)        
        decoder_output, _ = self.decoder(decoder_input)        
        predictions = self.output_layer(decoder_output)        
        return predictions