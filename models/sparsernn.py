import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Null(nn.Module):
    def __init__(self):
      super().__init__()
    
    def forward(self, x):
      return x

class SparseRNN(nn.Module):
  """

  input: a batched sequence of integer token ids.
  input shape : ( _ , sequence_length ).
  output_shape: ( _ , sequence_length, num_classes).


  For classification: index the last output embedding for the context vector.
  For POS tagging   : keep the output sequence as it is.
  """
  

  def __init__(self, sequence_length, embedding_dims, vocab_size, num_classes, hidden_state_sizes):
    super().__init__()
    self.sequence_length = sequence_length
    self.embedding_dims, vocab_size = embedding_dims, vocab_size
    self.embedder = nn.Embedding(vocab_size, embedding_dims, )
    self.num_classes = num_classes

    self.hidden_state_output_sizes = [self.embedding_dims] + hidden_state_sizes
    i = 0
    while 2**i + 1 < self.sequence_length:
      i += 1
    self.hidden_weights = []
    for prev_index, next_index in enumerate(range(1, len(self.hidden_state_output_sizes))):
      impulse_size = self.hidden_state_output_sizes[prev_index]
      output_size  = self.hidden_state_output_sizes[next_index]
      input_size   = impulse_size + i * output_size
      self.hidden_weights.append(
          nn.Sequential(
            nn.Linear(input_size, output_size),
            # nn.LayerNorm(output_size),
            nn.Tanh(),  
          )
      )
    self.hidden_weights = nn.ModuleList(self.hidden_weights)
    
    self.label_output_weights = nn.Sequential(
        nn.Linear(self.hidden_state_output_sizes[-1], self.num_classes)
    )

  def forward(self, x):
    cache = []
    x = self.embedder(x)
    batch_size = x.shape[0]
    # print(batch_size)
    hidden_input_stacks = []
    i = 0
    while 2**i + 1 < self.sequence_length:
      hidden_input_stacks.append(
          [
              torch.zeros(batch_size, input_size).to(device)
              for input_size in self.hidden_state_output_sizes[1:]
          ]
      )
      i += 1
    
    outputs = []
    for t in range(self.sequence_length):
      z = x[:, t, :]
      j = 0
      while 2**j <= len(cache):
        hidden_input_stacks[-(j+1)] = cache[-(2**j)]
        j += 1
      
      new_cache_stack = []
      for depth_index in range(len(self.hidden_weights)):
        hidden_input_line = []
        for hidden_input_stack in hidden_input_stacks:
          hidden_input_line.append(hidden_input_stack[depth_index])
        # print([hidden_input_line[idx].shape for idx in range(len(hidden_input_line))])
        # print(z.shape)
        z = torch.cat([z] + hidden_input_line, dim=1)
        z = self.hidden_weights[depth_index](z)
        new_cache_stack.append(z)
      cache.append(new_cache_stack)
      preds = self.label_output_weights(cache[-1][-1])
      outputs.append(preds)
    return torch.stack(outputs, dim=1)  # (batch_size, sequence_length, num_classes)
    
SparseRNN_kwargs = {
    'sequence_length': 28*28*1, ### edit this
    'embedding_dims': 64,   ### hyperparamater
    'vocab_size': 50000,   ### edit this
    'num_classes': 7,     ### edit this (or not)
    'hidden_state_sizes': [512, 128],  ### hyperparameter
}
# model = SparseRNN(**SparseRNN_kwargs).to(device)
# sequences = torch.ones(32, SparseRNN_kwargs['sequence_length']).type(torch.int64).to(device)
# print(model(sequences).shape)