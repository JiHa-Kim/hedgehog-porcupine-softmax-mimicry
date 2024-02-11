# Example for testing the softmax mimicry feature map
import torch
from softmax_mimicry_feature_map import SpikyMLPFeatureMap, attention_weight_distillation_loss

if __name__ == "__main__":
    # Assume some dimensions for input and output
    input_dim = 128  # Dimension of queries and keys
    output_dim = 256  # Dimension of the feature map

    # Create a batch of queries and keys
    batch_size = 32
    seq_length = 50
    queries = torch.randn(batch_size, seq_length, input_dim)
    keys = torch.randn(batch_size, seq_length, input_dim)

    # Initialize the spiky MLP feature map
    phi_mlp = SpikyMLPFeatureMap(input_dim, output_dim)

    # Compute the loss
    loss = attention_weight_distillation_loss(queries, keys, phi_mlp)
    print(f"Attention weight distillation loss: {loss.item()}")