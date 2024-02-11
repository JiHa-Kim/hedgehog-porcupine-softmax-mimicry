# Main implementation of the feature map used in the softmax mimicry
import torch
from torch import nn
from torch.nn import functional as F

class SpikyMLPFeatureMap(nn.Module):
    """
    A single-layer MLP with an exponential activation function to create a spiky feature map.
    
    Attributes:
        linear (nn.Linear): Linear transformation layer.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initializes the SpikyMLPFeatureMap.
        
        Args:
            input_dim (int): Dimension of the input vector.
            output_dim (int): Dimension of the output vector (feature map).
        """
        super(SpikyMLPFeatureMap, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass of the SpikyMLPFeatureMap.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying linear transformation and exponential activation.
        """
        linear_output = self.linear(x)
        spiky_feature_map = torch.exp(linear_output)
        return spiky_feature_map

def attention_weight_distillation_loss(queries, keys, phi_mlp):
    """
    Computes the attention weight distillation loss.
    
    Args:
        queries (torch.Tensor): Query vectors.
        keys (torch.Tensor): Key vectors.
        phi_mlp (SpikyMLPFeatureMap): Instance of SpikyMLPFeatureMap.
        
    Returns:
        torch.Tensor: The computed loss.
    """
    # Compute softmax attention weights
    softmax_weights = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)), dim=-1)
    
    # Compute spiky MLP attention weights
    phi_queries = phi_mlp(queries)
    phi_keys = phi_mlp(keys)
    spiky_mlp_weights = torch.matmul(phi_queries, phi_keys.transpose(-2, -1))
    spiky_mlp_weights = torch.softmax(spiky_mlp_weights, dim=-1)
    
    # Compute cross-entropy loss
    loss = F.kl_div(spiky_mlp_weights.log(), softmax_weights, reduction='batchmean')
    return loss
