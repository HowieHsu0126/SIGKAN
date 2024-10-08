from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Module, Parameter
from torch.nn.parameter import Parameter
import numpy as np
import math

class AttentionMechanism(torch.nn.Module):
    """
    Implements attention-based similarity calculation between nodes.
    """
    def __init__(self, in_features):
        super(AttentionMechanism, self).__init__()
        self.attn_fc = torch.nn.Linear(in_features, in_features)  # Linear transformation for input features
        self.out_proj = torch.nn.Linear(in_features, 1)  # Project down to similarity score

    def forward(self, x):
        """
        Compute attention-based similarity between nodes.

        Args:
            x (Tensor): Input node features of shape (batch_size, num_nodes, feature_dim).

        Returns:
            Tensor: Attention-based similarity matrix (batch_size, num_nodes, num_nodes).
        """
        # Compute attention scores
        batch_size, num_nodes, feature_dim = x.size()

        # Apply linear transformation on node features
        h = self.attn_fc(x)  # (batch_size, num_nodes, feature_dim)
        h = torch.tanh(h)  # Non-linearity to introduce interactions

        # Compute pairwise similarity using dot-product attention
        scores = torch.bmm(h, h.transpose(1, 2))  # (batch_size, num_nodes, num_nodes)

        # Apply projection to each node pair similarity score
        scores_flat = scores.view(batch_size * num_nodes, num_nodes)  # (batch_size * num_nodes, num_nodes)

        # The projection layer should be applied along the feature dimension (not num_nodes)
        h_proj = self.out_proj(h.view(batch_size * num_nodes, feature_dim))  # (batch_size * num_nodes, 1)

        # Reshape projected scores back to the original form
        attn_weights = h_proj.view(batch_size, num_nodes, 1)  # (batch_size, num_nodes, 1)

        # Apply softmax to get attention weights
        return F.softmax(attn_weights, dim=-1)  # Normalize to get attention weights




class BatchGraphConvolution(nn.Module):
    """
    Implements a batch version of the graph convolution layer, performing graph convolutions for each graph in a batch.

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        bias (bool): Whether to use a bias term in the layer.

    Attributes:
        weight (Parameter): The weight matrix for graph convolution.
        bias (Parameter or None): The bias term, if applicable.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix and bias
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        # Use Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x, lap):
        """
        Forward pass of the graph convolution layer.

        Args:
            x (Tensor): Node feature tensor of shape (batch_size, num_nodes, in_features).
            lap (Tensor): Laplacian matrix tensor of shape (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Output feature tensor of shape (batch_size, num_nodes, out_features).
        """
        batch_size = x.size(0)
        weight_expanded = self.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Perform graph convolution
        support = torch.bmm(x, weight_expanded)
        output = torch.bmm(lap, support)

        # Add bias if present
        if self.bias is not None:
            output += self.bias
        return output


class MultiHeadGraphAttention(nn.Module):
    """
    Implements multi-head graph attention layer for processing node features
    and adjacency matrices in graph neural networks.

    Attributes:
        n_head (int): Number of attention heads.
        w (Parameter): Weight matrix for input feature transformation, with size (n_head, f_in, f_out).
        a_src (Parameter): Weight matrix for attention mechanism on the source node, with size (n_head, f_out, 1).
        a_dst (Parameter): Weight matrix for attention mechanism on the destination node, with size (n_head, f_out, 1).
        leaky_relu (nn.LeakyReLU): Leaky ReLU activation function with a negative slope of 0.2.
        softmax (nn.Softmax): Softmax activation to normalize attention scores.
        dropout (nn.Dropout): Dropout layer for attention scores regularization.
        bias (Parameter, optional): Optional bias term.
    """

    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        """
        Initializes the multi-head attention layer.

        Args:
            n_head (int): Number of attention heads.
            f_in (int): Dimension of the input features.
            f_out (int): Dimension of the output features.
            attn_dropout (float): Dropout probability for attention scores.
            bias (bool): Whether to add a bias term (default: True).
        """
        super(MultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)

        # Optional bias initialization
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        # Initialize parameters with Xavier uniform distribution
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        """
        Forward pass for multi-head attention layer.

        Args:
            h (torch.Tensor): Node feature matrix of size (n, f_in).
            adj (torch.Tensor): Adjacency matrix of size (n, n).

        Returns:
            torch.Tensor: Updated node features after applying attention mechanism.
        """
        n = h.size(0)  # Number of nodes
        # Shape: (n_head, n, f_out)
        h_prime = torch.matmul(h.unsqueeze(0), self.w)

        # Compute attention scores for source and destination nodes
        attn_src = torch.bmm(h_prime, self.a_src)  # Shape: (n_head, n, 1)
        attn_dst = torch.bmm(h_prime, self.a_dst)  # Shape: (n_head, n, 1)

        # Combine source and destination attention scores
        # Shape: (n_head, n, n)
        attn = attn_src.expand(-1, -1, n) + \
            attn_dst.expand(-1, -1, n).permute(0, 2, 1)

        # Apply Leaky ReLU activation and mask attention
        attn = self.leaky_relu(attn)
        # Mask non-adjacent nodes
        attn.data.masked_fill_(1 - adj, float("-inf"))

        # Apply softmax normalization and dropout
        attn = self.softmax(attn)  # Shape: (n_head, n, n)
        attn = self.dropout(attn)

        # Update node features based on attention scores
        output = torch.bmm(attn, h_prime)  # Shape: (n_head, n, f_out)

        # Add bias if applicable
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class BatchMultiHeadGraphAttention(nn.Module):
    """
    Implements batch version of multi-head graph attention layer for batched node features and adjacency matrices.

    Args:
        n_head (int): Number of attention heads.
        f_in (int): Input feature dimension.
        f_out (int): Output feature dimension.
        attn_dropout (float): Dropout probability for attention scores.
        bias (bool): Whether to add a bias term.
    """

    def __init__(self, n_head, f_in, f_out, attn_dropout=0.0, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        # Activation, dropout, and optional bias
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.zeros_(self.bias)
        else:
            self.bias = None

        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        """
        Forward pass of batch multi-head graph attention layer.

        Args:
            h (Tensor): Node feature matrix of size (batch_size, num_nodes, f_in).
            adj (Tensor): Adjacency matrix of size (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Updated node features of size (batch_size, num_nodes, f_out).
        """
        bs, n = h.size()[:2]
        # Shape: (batch_size, n_head, num_nodes, f_out)
        h_prime = torch.matmul(h.unsqueeze(1), self.w)

        # Compute attention scores for source and destination nodes
        # Shape: (batch_size, n_head, num_nodes, 1)
        attn_src = torch.matmul(h_prime, self.a_src)
        # Shape: (batch_size, n_head, num_nodes, 1)
        attn_dst = torch.matmul(h_prime, self.a_dst)

        # Combine source and destination attention scores
        attn = attn_src.expand(-1, -1, -1, n) + \
            attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)
        attn = self.leaky_relu(attn)

        # Mask and apply softmax normalization
        # Shape: (batch_size, 1, num_nodes, num_nodes)
        mask = 1 - adj.unsqueeze(1)
        attn.data.masked_fill_(mask, float('-inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Compute updated node features
        # Shape: (batch_size, n_head, num_nodes, f_out)
        output = torch.matmul(attn, h_prime)

        # If bias is present, add it
        if self.bias is not None:
            output += self.bias

        return output

class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class SociallogicalInformedMessagePassing(nn.Module):
    """
    Implements the sociologically-informed message passing layer based on the bounded-confidence model (BCM).

    Args:
        in_features (int): Number of input features per node.
        out_features (int): Number of output features per node.
        delta (float): Bounded confidence interval for opinion dynamics.
        bias (bool): Whether to include a bias term. Default is True.
    """

    def __init__(self, in_features, out_features, delta=0.5, bias=True):
        super(SociallogicalInformedMessagePassing, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.delta = delta  # Bounded confidence interval

        # Learnable weight matrix for node features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        # KAN for updated node features
        self.updater = KANLinear(out_features, out_features)

    def forward(self, x, adj):
        """
        Forward pass for sociologically-informed message passing.

        Args:
            x (Tensor): Input node features of shape (batch_size, num_nodes, in_features).
            adj (Tensor): Ego adjacency matrix of shape (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Updated node features.
        """
        # Opinion difference: interaction term between node features
        batch_size, num_nodes, _ = x.size()
        # (batch_size, num_nodes, num_nodes, in_features)
        opinion_diff = x.unsqueeze(2) - x.unsqueeze(1)
        # (batch_size, num_nodes, num_nodes), compute distance
        opinion_diff = opinion_diff.norm(dim=-1)

        # Bounded confidence filter based on BCM
        # Binary mask for opinions within the confidence interval
        mask = (opinion_diff <= self.delta).float()

        # Apply adjacency and confidence interval to adjust influence
        adj_weighted = adj * mask  # Adjust adjacency with BCM filter
        # Normalize the weighted adjacency matrix to account for different neighborhood sizes
        degree = torch.sum(adj_weighted, dim=-1, keepdim=True) + 1e-6
        adj_weighted = adj_weighted / degree

        # Aggregate messages from neighbors
        aggregated_message = torch.bmm(adj_weighted, x)

        # Linear transformation with learned weight
        out = torch.bmm(aggregated_message, self.weight.unsqueeze(
            0).expand(batch_size, -1, -1))

        if self.bias is not None:
            out += self.bias

        # Apply non-linearity (e.g., ReLU) to introduce non-linearity to the model
        out = F.relu(out)

        # Pass the updated features through MLP
        out = self.updater(out)

        return out
    

