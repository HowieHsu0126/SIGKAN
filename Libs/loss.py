import torch
import torch.nn.functional as F

def psi(s_uv, epsilon_1=0.1, epsilon_2=0.5, mu=1.0, nu=-1.0):
    """
    Psi function to adjust the influence between nodes based on similarity s_uv.

    Args:
        s_uv (Tensor): Similarity between nodes u and v.
        epsilon_1 (float): Lower threshold for similarity.
        epsilon_2 (float): Upper threshold for similarity.
        mu (float): Amplification factor for high similarity.
        nu (float): Negative influence for dissimilar nodes.

    Returns:
        Tensor: Adjusted influence based on the similarity.
    """
    # Apply the piecewise function based on similarity thresholds
    psi_val = torch.where(s_uv > epsilon_2, mu * s_uv, 
                          torch.where(s_uv >= epsilon_1, s_uv, nu * (1 - s_uv)))
    return psi_val


def compute_ode_loss(model, features, vertices, adj, delta, psi):
    """
    Compute the ODE loss based on the difference between the LHS and RHS of the ODEs.
    
    Args:
        model (torch.nn.Module): The GNN model.
        features (Tensor): Input node features.
        adj (Tensor): Adjacency matrix.
        delta (float): Time step delta.
        psi (callable): The function that applies the psi transformation.
        
    Returns:
        Tensor: The computed ODE loss.
    """
    # Use automatic differentiation to compute the time derivative
    x_t = features.clone().detach().requires_grad_(True)
    x_next = model(x_t, vertices, adj)  # Next time step predicted by the model
    
    # Compute dx/dt (time derivative) using automatic differentiation
    time_derivative = torch.autograd.grad(outputs=x_next, inputs=x_t,
                                          grad_outputs=torch.ones_like(x_next),
                                          create_graph=True)[0]

    # Compute the similarity matrix s_uv (normalized adjacency matrix)
    degree = adj.sum(dim=-1, keepdim=True)
    adj_normalized = adj / (degree + 1e-6)  # Avoid division by zero

    # Apply psi(s_uv) to adjust the influence based on similarity
    rhs = torch.zeros_like(x_t)
    for u in range(x_t.size(1)):  # Loop over each node
        neighbors = adj[u].nonzero(as_tuple=True)[0]  # Find neighbors
        
        # Reshape x_t[u] for broadcasting
        x_u = x_t[:, u, :].unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)

        for v in neighbors:
            # Reshape x_t[v] for broadcasting
            x_v = x_t[:, v, :].unsqueeze(1)  # Shape: (batch_size, 1, feature_dim)
            
            s_uv = adj_normalized[u, v][0]  # Similarity between node u and v
            psi_val = psi(s_uv)  # Apply psi transformation
            
            # Calculate the difference and adjust with psi(s_uv)
            rhs += psi_val * (x_v - x_u)  # Adjusted difference

    # Compute the ODE loss as the MSE between the time derivative and the RHS of the ODE
    ode_loss = torch.mean((time_derivative - rhs) ** 2)
    return ode_loss



def compute_data_loss(output, labels):
    """
    Compute the data loss between predictions and ground truth.

    Args:
        output (Tensor): Model predictions.
        labels (Tensor): Ground truth labels.
    
    Returns:
        Tensor: Data loss.
    """
    return F.nll_loss(output, labels)


def generate_adversarial_perturbation(model, features, vertices, adj, epsilon, labels):
    """
    Generate adversarial perturbations for input node features.
    
    Args:
        model (torch.nn.Module): The GNN model.
        x (Tensor): Input node features.
        adj (Tensor): Adjacency matrix.
        epsilon (float): Perturbation budget.

    Returns:
        Tensor: Adversarial perturbations.
    """
    x_adv = features.clone().detach().requires_grad_(True)
    output = model(x_adv, vertices, adj)[:, -1, :]
    loss = F.nll_loss(output, labels)
    
    # Compute gradients with respect to x_adv
    gradients = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
    
    # Apply perturbations using the gradients (FGSM approach)
    perturbation = epsilon * gradients.sign()
    x_adv_perturbed = x_adv + perturbation
    
    return x_adv_perturbed



def compute_total_loss(model, features, vertices, adj, labels, alpha, beta, delta, epsilon, psi):
    """
    Compute the total loss which includes data loss, ODE loss, and adversarial loss.
    
    Args:
        model (torch.nn.Module): The GNN model.
        features (Tensor): Input features.
        vertices (Tensor): Input vertices.
        adj (Tensor): Adjacency matrix.
        labels (Tensor): Ground truth labels.
        alpha (float): Weight of ODE loss.
        beta (float): Weight of adversarial loss.
        delta (float): Time step delta for ODE loss.
        epsilon (float): Perturbation budget for adversarial loss.
        psi (callable): The function that applies the psi transformation.
        
    Returns:
        Tensor: The computed total loss.
    """
    # Data loss
    output = model(features, vertices, adj)[:, -1, :]
    data_loss = F.nll_loss(output, labels)

    # ODE loss
    ode_loss = compute_ode_loss(model, features, vertices, adj, delta, psi)

    # Adversarial loss
    x_adv = generate_adversarial_perturbation(model, features, vertices, adj, epsilon, labels)
    output_adv = model(x_adv, vertices, adj)[:, -1, :]
    adversarial_loss = F.nll_loss(output_adv, labels)

    # Total loss
    total_loss = data_loss + alpha * ode_loss + beta * adversarial_loss
    return total_loss

