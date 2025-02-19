import torch
import torch.nn as nn
import numpy as np
import random

class binary_disordered_RNNwavefunction(nn.Module):
    def __init__(self, key, system_size, inputdim, nlayers, activation="relu",units=10, seed=111, type=torch.float32, device='cpu'):
        super(binary_disordered_RNNwavefunction, self).__init__()
        
        self.hidden_dim = units
        self.input_dim = inputdim       # e.g., 2 for binary (0/1)
        self.n_layers = nlayers
        self.N = system_size             # Total number of sites (spins)
        self.activation = str(activation)
        self.key = key
        self.type = type
        self.device = device

        # Permutation of sites (by default, ordered 0,1,...,N-1)
        self.permuted = np.arange(self.N)

        # Create a separate RNN cell (or set of cells) for each site.
        if self.key == "vanilla":
            rnn_cells = []
            for _ in range(self.N):
                input_size = self.input_dim
                for _ in range(self.n_layers):
                    rnn_cells.append(nn.RNNCell(input_size=input_size,hidden_size=self.hidden_dim, nonlinearity=self.activation,dtype=self.type))
                    input_size = self.hidden_dim  # Update for next layer in the stack
            self.rnn = nn.ModuleList(rnn_cells).to(self.device)
            
        elif self.key == "gru":
            rnn_cells = []
            for _ in range(self.N):
                input_size = self.input_dim
                for _ in range(self.n_layers):
                    rnn_cells.append(nn.GRUCell(input_size=input_size,hidden_size=self.hidden_dim,dtype=self.type))
                    input_size = self.hidden_dim
            self.rnn = nn.ModuleList(rnn_cells).to(self.device)

        # Create a dense (output) network for each site.
        # This network maps the RNN's hidden state to a probability distribution over outputs.
        dense = []
        for _ in range(self.N):
            dense.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.type),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.input_dim, dtype=self.type),
                nn.Softmax(dim=1)
            ))
        self.dense = nn.ModuleList(dense).to(self.device)

    def forward(self, inputs):

        hiddens = []  # List to store hidden states per RNN layer
        batch_size = inputs.shape[0]

        #rnn_state = torch.zeros(batch_size, self.hidden_dim, dtype=self.type).to(self.device)
        rnn_state_zero = torch.zeros(batch_size, self.hidden_dim, dtype=self.type).to(self.device)
        
        probs = torch.zeros(batch_size, self.N, self.input_dim, dtype=self.type).to(self.device)
        probs[:, :, 1] = 1.0

        self.samples = torch.zeros(batch_size, self.N, dtype=torch.long).to(self.device)

        # Process each site in the given (or permuted) order.
        for site in self.permuted:
            if site == self.permuted[0]:
                # For the first site, run through each RNN layer using the initial input.
                for layer in range(self.n_layers):
                    rnn_state = self.rnn[layer](inputs, rnn_state_zero)
                    inputs = rnn_state
                    hiddens.append(rnn_state)
            else:
                # For subsequent sites, reuse the hidden state from the corresponding layer.
                for layer in range(self.n_layers):
                    indice = layer + site * self.n_layers  # Each site has its own set of layers.
                    rnn_state = hiddens[layer]
                    rnn_state = self.rnn[indice](inputs, rnn_state)
                    inputs = rnn_state
                    hiddens[layer] = rnn_state

            # Apply a cosine transformation to the final hidden state.
            #rnn_state = torch.cos(rnn_state) + 1e-10

            # Pass the transformed state through the site's dense network to get output probabilities.
            logits = self.dense[site](rnn_state)
            probs[:, site, :] = logits + 1e-10

            # Sample from the output distribution.
            sample = torch.reshape(torch.multinomial(logits + 1e-10, 1), (-1,))
            self.samples[:, site] = sample

            # Prepare the one-hot encoded sample to serve as the input for the next site.
            inputs = torch.nn.functional.one_hot(sample, num_classes=self.input_dim).type(self.type)

        # Compute the log-probability of the generated sequence.
        one_hot_samples = torch.nn.functional.one_hot(self.samples, num_classes=self.input_dim).type(self.type)
        self.log_probs = torch.sum(torch.log(torch.sum(torch.multiply(probs, one_hot_samples), dim=2) + 1e-10),dim=1)

        return probs


