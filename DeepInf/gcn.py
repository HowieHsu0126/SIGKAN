import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layers import BatchGraphConvolution

class BatchGCN(nn.Module):
    """
    BatchGCN: A multi-layer Graph Convolutional Network (GCN) with optional instance 
    normalization and vertex features for batch processing.

    Args:
        n_units (list[int]): Number of units in each GCN layer. The first unit corresponds 
                             to the input dimension, and the last corresponds to the output.
        dropout (float): Dropout probability for regularization.
        pretrained_emb (torch.Tensor): Pretrained embedding tensor for nodes.
        vertex_feature (torch.Tensor): Optional tensor for additional vertex features.
        use_vertex_feature (bool): Whether to include vertex features in the input.
        fine_tune (bool, optional): If True, fine-tune the pretrained embeddings. Defaults to False.
        instance_normalization (bool, optional): If True, applies instance normalization to embeddings.
                                                 Defaults to False.
    """
    def __init__(
        self,
        n_units: list[int],
        dropout: float,
        pretrained_emb: torch.Tensor,
        vertex_feature: torch.Tensor = None,
        use_vertex_feature: bool = False,
        fine_tune: bool = False,
        instance_normalization: bool = False,
    ):
        super(BatchGCN, self).__init__()

        # Initialize parameters
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.use_vertex_feature = use_vertex_feature
        self.inst_norm = instance_normalization

        # Embedding layer with optional fine-tuning
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=not fine_tune)

        # Adjust input dimension based on embedding size
        n_units[0] += pretrained_emb.size(1)

        # Optional vertex feature embedding
        if self.use_vertex_feature and vertex_feature is not None:
            self.vertex_feature = nn.Embedding.from_pretrained(vertex_feature, freeze=True)
            n_units[0] += vertex_feature.size(1)

        # Optional instance normalization layer
        self.norm = (
            nn.InstanceNorm1d(pretrained_emb.size(1), momentum=0.0, affine=True)
            if self.inst_norm
            else None
        )

        # GCN layers defined as a stack of BatchGraphConvolution layers
        self.layer_stack = nn.ModuleList(
            [BatchGraphConvolution(n_units[i], n_units[i + 1]) for i in range(self.num_layer)]
        )

    def forward(self, x: torch.Tensor, vertices: torch.Tensor, lap: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BatchGCN model.

        Args:
            x (torch.Tensor): Input feature tensor (batch_size, num_vertices, input_dim).
            vertices (torch.Tensor): Vertex indices for embedding lookup (batch_size, num_vertices).
            lap (torch.Tensor): Laplacian matrix for GCN layer propagation.

        Returns:
            torch.Tensor: Log-softmax output over the final layer.
        """
        # Retrieve embeddings for vertices
        emb = self.embedding(vertices)

        # Apply instance normalization if enabled
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)

        # Concatenate input features with embeddings
        x = torch.cat((x, emb), dim=2)

        # Concatenate additional vertex features if used
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)

        # Pass through GCN layers with activation and dropout
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)  # GCN propagation
            if i < self.num_layer - 1:
                x = F.elu(x)  # Non-linear activation (Exponential Linear Unit)
                x = F.dropout(x, self.dropout, training=self.training)  # Dropout regularization

        return F.log_softmax(x, dim=-1)  # Log-softmax for classification output


# Example usage:
# model = BatchGCN(n_units=[64, 32, 16], dropout=0.5, pretrained_emb=pretrained_emb_tensor, 
#                  vertex_feature=vertex_feature_tensor, use_vertex_feature=True)
