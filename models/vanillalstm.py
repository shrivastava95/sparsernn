import torch
from torch import nn


from models.sparsernn import Null

class EncoderLSTM(nn.Module):
    def __init__(self, embedding_dims:int, vocab_size:int, num_classes:int, hidden_state_sizes):
        super().__init__()
        num_layers = len(hidden_state_sizes)
        self.num_layers = num_layers
        hidden_layer_size = hidden_state_sizes[0]
        self.get_embeddings = nn.Embedding(vocab_size, embedding_dims)
        self.rnn = nn.LSTM(input_size=embedding_dims, hidden_size=hidden_layer_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_layer_size, num_classes),
        )
        self.dropout = Null()
        self.h0 = nn.Parameter(torch.randn(self.num_layers, 1, hidden_layer_size))
        self.c0 = nn.Parameter(torch.randn(self.num_layers, 1, hidden_layer_size))

    
    def forward(self, input_sequence, hidden_states=None):
        embeddings = self.dropout(self.get_embeddings(input_sequence))
        if hidden_states == None:
          hidden_states = (
              self.h0.expand(-1, input_sequence.size(0), -1).contiguous(),
              self.c0.expand(-1, input_sequence.size(0), -1).contiguous(),
          )
        outputs, hidden_states = self.rnn(embeddings, hidden_states)
        outputs = torch.cat([self.fc(outputs[:, index]).unsqueeze(1) for index in range(outputs.shape[1])], dim=1)
        return outputs