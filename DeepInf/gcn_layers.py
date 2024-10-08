from __future__ import absolute_import, unicode_literals, division, print_function
import torch
from torch.nn import Module, Parameter
import torch.nn.init as init

class BatchGraphConvolution(Module):
    """
    Implements a batch version of the graph convolution layer.
    
    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        bias (bool): If True, adds a learnable bias to the output. Defaults to True.
        
    Attributes:
        weight (Parameter): Learnable weight matrix of shape (in_features, out_features).
        bias (Parameter or None): Learnable bias vector of shape (out_features) or None if no bias is used.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Define weight as a learnable parameter and initialize it
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        init.xavier_uniform_(self.weight)  # Xavier initialization for weights

        # Define bias if required
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)  # Initialize bias to zero
        else:
            self.register_parameter('bias', None)  # Register None for no bias

    def forward(self, x, lap):
        """
        Forward pass for graph convolution.
        
        Args:
            x (Tensor): Input feature tensor of shape (batch_size, num_nodes, in_features).
            lap (Tensor): Laplacian matrix tensor of shape (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Output feature tensor of shape (batch_size, num_nodes, out_features).
        """
        # Expand the weight tensor to match the batch size
        expand_weight = self.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Perform matrix multiplication of input and weight
        support = torch.bmm(x, expand_weight)
        
        # Multiply the result by the Laplacian matrix
        output = torch.bmm(lap, support)
        
        # Add bias if it exists
        if self.bias is not None:
            return output + self.bias
        else:
            return output
