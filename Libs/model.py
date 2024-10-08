import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import BatchGraphConvolution, SociallogicalInformedMessagePassing
from torch.nn import ModuleList


class BatchGCN(nn.Module):
    """
    BatchGCN: Multi-layer GCN supporting batch processing, vertex features, 
    optional instance normalization, and pre-trained embeddings.

    Args:
        n_units (list[int]): Number of units per GCN layer.
        dropout (float): Dropout rate.
        pretrained_emb (Tensor): Pre-trained node embeddings.
        vertex_feature (Tensor, optional): Additional vertex features.
        use_vertex_feature (bool): Whether to include vertex features in input.
        fine_tune (bool, optional): Whether to fine-tune pre-trained embeddings.
        instance_normalization (bool, optional): Whether to use instance normalization.
    """

    def __init__(self, n_units, dropout, pretrained_emb, vertex_feature=None,
                 use_vertex_feature=False, fine_tune=False, instance_normalization=False):
        super(BatchGCN, self).__init__()

        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.use_vertex_feature = use_vertex_feature
        self.inst_norm = instance_normalization

        # Embedding layer for nodes
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_emb, freeze=not fine_tune)

        # Adjust input dimensions to include embedding size
        n_units[0] += pretrained_emb.size(1)

        # Optional vertex features
        if use_vertex_feature and vertex_feature is not None:
            self.vertex_feature = nn.Embedding.from_pretrained(
                vertex_feature, freeze=True)
            n_units[0] += vertex_feature.size(1)

        # Optional instance normalization
        self.norm = nn.InstanceNorm1d(pretrained_emb.size(
            1), momentum=0.0, affine=True) if instance_normalization else None

        # GCN layers
        self.layer_stack = nn.ModuleList([BatchGraphConvolution(
            n_units[i], n_units[i + 1]) for i in range(self.num_layer)])

    def forward(self, x, vertices, lap):
        """
        Forward pass of BatchGCN.

        Args:
            x (Tensor): Input feature tensor (batch_size, num_vertices, input_dim).
            vertices (Tensor): Vertex indices for embeddings (batch_size, num_vertices).
            lap (Tensor): Laplacian matrix for GCN propagation.

        Returns:
            Tensor: Log-softmax output (batch_size, num_vertices, num_classes).
        """
        # Retrieve embeddings for vertices
        emb = self.embedding(vertices)

        # Apply instance normalization if applicable
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)

        # Concatenate input features with embeddings
        x = torch.cat((x, emb), dim=2)

        # Add vertex features if available
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)

        # Pass through GCN layers
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i < self.num_layer - 1:
                x = F.elu(x)  # Apply activation
                x = F.dropout(x, self.dropout, training=self.training)

        # Output log-softmax for classification
        return F.log_softmax(x, dim=-1)


class BatchGAT(nn.Module):
    """ 
    Batch Graph Attention Network (GAT) module for graph-structured data.

    This class implements a multi-layer GAT with optional vertex feature embeddings, 
    instance normalization, and dropout.

    Attributes:
        n_layer (int): Number of GAT layers.
        dropout (float): Dropout probability for the non-attention part of the GAT.
        inst_norm (bool): Whether to use instance normalization.
        embedding (nn.Embedding): Embedding layer initialized with pretrained word embeddings.
        vertex_feature (nn.Embedding, optional): Embedding for vertex features.
        layer_stack (nn.ModuleList): Stack of GAT layers, each layer is a multi-head attention layer.
    """

    def __init__(self, pretrained_emb, vertex_feature, use_vertex_feature,
                 n_units=[1433, 8, 7], n_heads=[8, 1],
                 dropout=0.1, attn_dropout=0.0, fine_tune=False,
                 instance_normalization=False):
        """
        Initialize the BatchGAT model.

        Args:
            pretrained_emb (torch.Tensor): Pretrained word embedding weights.
            vertex_feature (torch.Tensor): Pretrained vertex feature embeddings.
            use_vertex_feature (bool): Flag to include vertex features in the model.
            n_units (list): Number of units (neurons) in each GAT layer.
            n_heads (list): Number of attention heads per GAT layer.
            dropout (float): Dropout probability for non-attention layers.
            attn_dropout (float): Dropout probability for attention heads.
            fine_tune (bool): Whether to fine-tune the pretrained embeddings.
            instance_normalization (bool): Whether to apply instance normalization.
        """
        super(BatchGAT, self).__init__()

        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization

        # Instance normalization if enabled
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(
                pretrained_emb.size(1), momentum=0.0, affine=True)

        # Embedding layer with pretrained embeddings
        self.embedding = nn.Embedding(
            pretrained_emb.size(0), pretrained_emb.size(1))
        self.embedding.weight = nn.Parameter(pretrained_emb)
        self.embedding.weight.requires_grad = fine_tune

        # Update input units based on embedding dimension
        n_units[0] += pretrained_emb.size(1)

        self.use_vertex_feature = use_vertex_feature

        # Optional vertex feature embeddings
        if self.use_vertex_feature:
            self.vertex_feature = nn.Embedding(
                vertex_feature.size(0), vertex_feature.size(1))
            self.vertex_feature.weight = nn.Parameter(vertex_feature)
            self.vertex_feature.weight.requires_grad = False
            n_units[0] += vertex_feature.size(1)

        # Create a stack of GAT layers
        self.layer_stack = nn.ModuleList()
        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i > 0 else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                                             f_out=n_units[i + 1], attn_dropout=attn_dropout)
            )

    def forward(self, x, vertices, adj):
        """
        Forward pass through the BatchGAT model.

        Args:
            x (torch.Tensor): Input node features of shape (batch_size, num_nodes, feature_dim).
            vertices (torch.Tensor): Vertex IDs corresponding to the input nodes.
            adj (torch.Tensor): Adjacency matrix of the graph.

        Returns:
            torch.Tensor: Log-softmax of the final layer output.
        """
        # Apply word embeddings to the vertices
        emb = self.embedding(vertices)

        # Apply instance normalization if enabled
        if self.inst_norm:
            emb = self.norm(emb.transpose(1, 2)).transpose(1, 2)

        # Concatenate input features with the embeddings
        x = torch.cat((x, emb), dim=2)

        # Concatenate vertex features if used
        if self.use_vertex_feature:
            vfeature = self.vertex_feature(vertices)
            x = torch.cat((x, vfeature), dim=2)

        # Process through GAT layers
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj)  # Apply GAT layer
            if i + 1 == self.n_layer:
                # Average over attention heads in the final layer
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(
                    bs, n, -1))  # Apply ELU activation
                # Apply dropout
                x = F.dropout(x, self.dropout, training=self.training)

        # Return the log-softmax over the last dimension
        return F.log_softmax(x, dim=-1)


class SIGKANBase(torch.nn.Module):
    """
    Base class for SIGKAN models that implement sociologically-informed message-passing.

    Args:
        n_units (list[int]): Number of units in each layer.
        delta (float): Bounded confidence interval for the BCM.
        dropout (float): Dropout rate.
    """

    def __init__(self, n_units, delta, dropout):
        super(SIGKANBase, self).__init__()
        self.num_layers = len(n_units) - 1
        self.delta = delta
        self.dropout = dropout

        # Define a stack of sociologically-informed message-passing layers
        self.layers = ModuleList([SociallogicalInformedMessagePassing(n_units[i], n_units[i + 1], delta)
                                  for i in range(self.num_layers)])

    def forward(self, x, adj):
        """
        Forward pass for SIGKAN models.

        Args:
            x (Tensor): Input node features (batch_size, num_nodes, in_features).
            adj (Tensor): Ego adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Updated node features after applying the layers.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=-1)


class SIGKAN_Norm(SIGKANBase):
    """
    SIGKAN_Norm: Uses the normalized adjacency matrix as the similarity function s_uv.
    """

    def __init__(self, n_units, delta, dropout):
        super(SIGKAN_Norm, self).__init__(n_units, delta, dropout)

    def normalize_adj(self, adj):
        """
        Normalize the adjacency matrix using the symmetric normalization technique.

        Args:
            adj (Tensor): Unnormalized adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Normalized adjacency matrix.
        """
        # Add self-loops by adding identity to the adjacency matrix
        adj = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)

        # Compute degree matrix D
        degree = adj.sum(dim=-1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float(
            'inf')] = 0  # Avoid division by zero

        # Perform symmetric normalization: D^{-1/2} A D^{-1/2}
        adj_normalized = degree_inv_sqrt.unsqueeze(
            -1) * adj * degree_inv_sqrt.unsqueeze(-2)

        return adj_normalized

    def forward(self, x, vertices, adj):
        """
        Forward pass for SIGKAN_Norm using normalized adjacency matrix similarity.

        Args:
            x (Tensor): Input node features (batch_size, num_nodes, feature_dim).
            vertices (Tensor): Ego vertex IDs corresponding to the input nodes.
            adj (Tensor): Unnormalized adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Updated node features.
        """
        # Normalize the adjacency matrix
        adj_normalized = self.normalize_adj(adj)

        # Proceed with the forward pass using the normalized adjacency matrix
        return super(SIGKAN_Norm, self).forward(x, adj_normalized)


class SIGKAN_Att(SIGKANBase):
    """
    SIGKAN_Att: Uses attention-based similarity computed from node features.
    """

    def __init__(self, n_units, delta, dropout, attention):
        super(SIGKAN_Att, self).__init__(n_units, delta, dropout)
        self.attention = attention  # Attention mechanism to compute s_uv

    def forward(self, x, vertices, adj):
        """
        Forward pass for SIGKAN_Att using attention-based similarity.

        Args:
            x (Tensor): Input node features (batch_size, num_nodes, feature_dim).
            vertices (Tensor): Ego vertex IDs corresponding to the input nodes.
            adj (Tensor): Adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Tensor: Updated node features.
        """
        # Compute attention-based similarity from node features
        # s_uv shape: (batch_size, num_nodes, num_nodes)
        s_uv = self.attention(x)
        weighted_adj = adj * s_uv  # Adjust adjacency matrix with attention-based similarity

        return super(SIGKAN_Att, self).forward(x, weighted_adj)
