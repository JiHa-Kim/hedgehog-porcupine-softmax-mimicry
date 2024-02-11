import unittest
import torch
from source.softmax_mimicry_feature_map import SpikyMLPFeatureMap, attention_weight_distillation_loss

class TestSpikyMLPFeatureMap(unittest.TestCase):
    
    def test_forward(self):
        input_dim = 128
        output_dim = 256
        batch_size = 32
        seq_length = 50
        input_tensor = torch.randn(batch_size, seq_length, input_dim)
        feature_map = SpikyMLPFeatureMap(input_dim, output_dim)
        output = feature_map(input_tensor)
        # Add assertions to validate the output
    
    def test_attention_weight_distillation_loss(self):
        input_dim = 128
        batch_size = 32
        seq_length = 50
        queries = torch.randn(batch_size, seq_length, input_dim)
        keys = torch.randn(batch_size, seq_length, input_dim)
        phi_mlp = SpikyMLPFeatureMap(input_dim, input_dim)  # Adjust output_dim as needed
        loss = attention_weight_distillation_loss(queries, keys, phi_mlp)
        # Add assertions to validate the loss

if __name__ == '__main__':
    unittest.main()