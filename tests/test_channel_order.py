import unittest

import torch
from torch import nn

from layers.CMIE_layer import DataMerge


class _EchoDecoder(nn.Module):
    def forward(self, target, memory):
        return target


class _ZeroModule(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class DataMergeChannelOrderTest(unittest.TestCase):
    def test_grouped_outputs_are_restored_to_original_channel_positions(self):
        merge = DataMerge(device="cpu", seq_len=2, num_heads=1, num_layers=1)
        merge.transformer_decoder = _EchoDecoder()
        merge.linear = _ZeroModule()
        merge.linear2 = _ZeroModule()

        x = torch.tensor(
            [[[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]]]
        )
        patch_assign = torch.tensor([1, 0, 1, 0])

        output = merge(x, patch_assign)

        self.assertTrue(torch.equal(output, 2 * x))


if __name__ == "__main__":
    unittest.main()
