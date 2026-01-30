import unittest

import torch

from model import MCformer


class MCformerConfigs:
    seq_len = 128
    n_classes = 11
    d_model = 64
    d_ff = 256
    n_heads = 8
    n_layers = 4
    dropout = 0.1


class TestModel(unittest.TestCase):
    """The the baseline model for Auto Modulation Classification"""

    inputs = torch.rand((4, 2, 128))

    def test_MCformer(self) -> None:
        """Test the MCformer model"""

        # Create the model
        model = MCformer.Model(configs=MCformerConfigs())

        # Do forward pass
        outputs = model(self.inputs)

        # Check the output shape
        self.assertEqual(outputs.shape, (4, 11))
