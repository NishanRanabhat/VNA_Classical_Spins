import torch
import torch.nn as nn
import numpy as np
import random

class binary_disordered_RNNwavefunction(nn.Module):

    """
    PyTorch module representing a binary disordered RNN wavefunction.

    This neural network architecture is designed to model the wavefunction
    of a binary disordered system (e.g., spins with two possible states).
    The model utilizes recurrent neural network (RNN) cells (either vanilla RNN
    or GRU) arranged per site, with a dedicated dense network mapping the RNN's
    hidden state to a probability distribution over binary outputs. The network
    sequentially processes each site (spin) and samples an output according to the
    learned probabilities, thereby generating a configuration and computing its
    log-probability.

    Attributes:
        hidden_dim (int): Number of units in the hidden layer for each RNN cell.
        input_dim (int): Dimensionality of the input, e.g., 2 for binary (0/1) inputs.
        n_layers (int): Number of stacked RNN layers per site.
        N (int): Total number of sites (spins) in the system.
        activation (str): Activation function used in vanilla RNN cells (e.g., "relu").
        key (str): Type of RNN cell to use; options are "vanilla" or "gru".
        type (torch.dtype): Data type for the model's parameters.
        device (str): Device on which the model will be run (e.g., 'cpu' or 'cuda').
        permuted (np.ndarray): Array of site indices; by default, sites are ordered sequentially.
        rnn (nn.ModuleList): List of RNN or GRU cells, arranged per site.
        dense (nn.ModuleList): List of dense networks (one per site) that map the hidden state
                               to output probabilities.
        samples (torch.Tensor): Tensor to store the sampled output configuration.
        log_probs (torch.Tensor): Tensor storing the log-probability of the generated sequence.

    Parameters:
        key (str): Specifies the type of RNN cell to use ("vanilla" for nn.RNNCell,
                   "gru" for nn.GRUCell).
        system_size (int): Total number of sites (spins) in the system.
        inputdim (int): Dimensionality of the input at each site.
        nlayers (int): Number of recurrent layers to stack per site.
        activation (str, optional): Activation function for the vanilla RNN. Default is "relu".
        units (int, optional): Number of units in each RNN cell. Default is 10.
        seed (int, optional): Random seed for reproducibility. Default is 111.
        type (torch.dtype, optional): Data type for tensors. Default is torch.float32.
        device (str, optional): Device to run the model on. Default is 'cpu'.

    Example:
        >>> model = binary_disordered_RNNwavefunction(
                key="gru", system_size=20, inputdim=2, nlayers=2, activation="relu", units=50, device="cuda"
            )
        >>> inputs = torch.ones(5, 2)  # batch of 5 samples with one-hot binary inputs
        >>> output_probs = model(inputs)
    """

    def __init__(self, key, system_size, inputdim, nlayers, activation="relu",units=10, type=torch.float32, device='cpu'):
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

        """
        Perform a forward pass through the network to generate a wavefunction sample.

        The forward pass processes the input sequentially for each site (spin) in the
        system. For each site, the network:
            1. Propagates the input through its stack of RNN cells.
            2. Uses a dense network to compute the probability distribution over
               the binary outputs.
            3. Samples from the computed distribution to obtain a configuration.
            4. Feeds a one-hot encoded version of the sample as input for the next site.
        After processing all sites, the method computes the total log-probability
        of the generated configuration.

        Parameters:
            inputs (torch.Tensor): The initial input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: A tensor of probabilities with shape (batch_size, N, input_dim),
                          where each entry corresponds to the output probabilities at a site.
        """

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

class binary_disordered_RNNwavefunction_weight_sharing(nn.Module):
    """
    PyTorch module representing a binary disordered RNN wavefunction with weight sharing.
    
    This architecture uses a single set of RNN layers (here GRU cells) and a single dense network
    that are applied identically at every site. This enforces weight sharing across sites, ensuring 
    that the same transformation is applied regardless of site index.
    
    Attributes:
        hidden_dim (int): Number of units in the hidden layer for each RNN cell.
        input_dim (int): Dimensionality of the input, e.g., 2 for binary (0/1) inputs.
        n_layers (int): Number of stacked RNN layers.
        N (int): Total number of sites (spins) in the system.
        key (str): Type of RNN cell to use; here we implement "gru" (other types can be added).
        type (torch.dtype): Data type for the model's parameters.
        device (str): Device on which the model will be run (e.g., 'cpu' or 'cuda').
        rnn_layers (ModuleList): Shared GRUCell layers used for every site.
        dense_network (nn.Sequential): Shared dense network that maps the final hidden state 
                                       to a probability distribution over outputs.
    """

    def __init__(self, key, system_size, inputdim, nlayers, activation="relu", units=10, type=torch.float32, device='cpu'):
        super(binary_disordered_RNNwavefunction_weight_sharing, self).__init__()

        self.hidden_dim = units
        self.input_dim = inputdim       # e.g., 2 for binary (0/1)
        self.n_layers = nlayers
        self.N = system_size  
        self.activation = str(activation)           # Total number of sites (spins)
        self.key = key
        self.type = type
        self.device = device

        # Create a single stack of RNN layers to be used for all sites.
        if self.key == "gru":
            self.rnn_layers = nn.ModuleList([
                nn.GRUCell(input_size=self.input_dim if i == 0 else self.hidden_dim,
                           hidden_size=self.hidden_dim,
                           dtype=self.type)
                for i in range(self.n_layers)
            ]).to(self.device)
        else:
            # Fallback for vanilla RNN; similar construction.
            self.rnn_layers = nn.ModuleList([
                nn.RNNCell(input_size=self.input_dim if i == 0 else self.hidden_dim,
                           hidden_size=self.hidden_dim,
                           nonlinearity=activation,
                           dtype=self.type)
                for i in range(self.n_layers)
            ]).to(self.device)

        # Create a single dense (output) network to be applied at every site.
        self.dense_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.type),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim, dtype=self.type),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, inputs):
        """
        Perform a forward pass to generate a wavefunction sample.
        
        The model sequentially processes each site using the same RNN layers and dense network.
        At each site:
          1. The input is passed through the shared stack of RNN layers.
          2. The final hidden state is fed into the shared dense network to obtain output probabilities.
          3. A binary sample is drawn from this distribution.
          4. The sampled one-hot vector is used as input for the next site.
          
        Returns:
            probs (torch.Tensor): Tensor of probabilities with shape (batch_size, N, input_dim).
                                  This stores the output probabilities for each site.
        """
        batch_size = inputs.shape[0]
        # Initialize hidden states for each RNN layer.
        h = [torch.zeros(batch_size, self.hidden_dim, dtype=self.type, device=self.device)
             for _ in range(self.n_layers)]

        # Container for probabilities at each site.
        probs = torch.zeros(batch_size, self.N, self.input_dim, dtype=self.type, device=self.device)
        # Container for sampled configurations.
        samples = torch.zeros(batch_size, self.N, dtype=torch.long, device=self.device)

        # Process each site sequentially.
        for site in range(self.N):
            x = inputs
            # Pass through the shared stack of RNN layers.
            for layer in range(self.n_layers):
                h[layer] = self.rnn_layers[layer](x, h[layer])
                x = h[layer]

            # Use the dense network to get probabilities.
            logits = self.dense_network(x)
            probs[:, site, :] = logits + 1e-10  # avoid numerical issues
            
            # Sample from the probability distribution.
            sample = torch.multinomial(logits + 1e-10, num_samples=1).view(-1)
            samples[:, site] = sample
            
            # Prepare the one-hot encoded sample as input for the next site.
            inputs = torch.nn.functional.one_hot(sample, num_classes=self.input_dim).type(self.type)

        # Compute the log-probability of the generated sequence.
        one_hot_samples = torch.nn.functional.one_hot(samples, num_classes=self.input_dim).type(self.type)
        log_probs = torch.sum(torch.log(torch.sum(probs * one_hot_samples, dim=2) + 1e-10), dim=1)
        
        # Store samples and log probabilities as attributes.
        self.samples = samples
        self.log_probs = log_probs

        return probs
