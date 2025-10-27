"""
trans_v5_2.py
A robust visual encoder for RL training.

Replaces the patch-embedding style with a straightforward Conv -> GroupNorm -> activation stack,
followed by a small MLP projection. GroupNorm/LayerNorm are used so the encoder behaves well
with small/variable minibatches typical in RL.

Class: NatureVisualEncoder(height, width, initial_channels, output_size)
Returns a tensor of shape [batch, output_size].
"""

# import torch
# import torch.nn as nn
# from mlagents.torch_utils import Initialization
# from mlagents.trainers.torch.layers import linear_layer
# from mlagents.trainers.torch import exporting_to_onnx





class NatureVisualEncoder(nn.Module):
    """Conv-based visual encoder designed for RL.

    Args:
        height (int): input image height
        width (int): input image width
        initial_channels (int): number of channels in input images
        output_size (int): final embedding size (h_size) returned by forward
    """

    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size

        # Convolutional trunk. Use relatively small filters with strides to reduce spatial size.
        # GroupNorm is used because it's robust to small minibatch sizes in RL.
        self.conv1 = nn.Conv2d(initial_channels, 32, kernel_size=8, stride=4, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(8, 32)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(8, 64)
        self.act2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.gn3 = nn.GroupNorm(8, 64)
        self.act3 = nn.LeakyReLU()

        # Compute final flat size after the conv stack
        conv1_hw = self.conv_output_shape((height, width), 8, 4)
        conv2_hw = self.conv_output_shape(conv1_hw, 4, 2)
        conv3_hw = self.conv_output_shape(conv2_hw, 3, 1)
        final_flat = conv3_hw[0] * conv3_hw[1] * 64
        self.final_flat = final_flat

        # Small projection head: linear -> layernorm -> activation
        # Use the project's linear_layer to follow initialization conventions.
        self.fc = nn.Sequential(
            linear_layer(self.final_flat, self.h_size, kernel_init=Initialization.KaimingHeNormal, kernel_gain=1.41),
            nn.LayerNorm(self.h_size),
            nn.LeakyReLU(),
        )

    @staticmethod
    def conv_output_shape(hw, kernel, stride, padding=0):
        """Compute (h,w) after a Conv2d with given kernel/stride/padding (integer values)."""
        h, w = hw
        h = (h + 2 * padding - kernel) // stride + 1
        w = (w + 2 * padding - kernel) // stride + 1
        return (h, w)
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        """Forward the visual observation through the encoder.

        visual_obs: expected shape [B, H, W, C] (ML-Agents convention). Returns [B, h_size].
        """
        # ML-Agents uses NHWC by default in some pipelines; convert to NCHW unless exporting to ONNX
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])

        x = self.conv1(visual_obs)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.act3(x)

        x = x.reshape(-1, self.final_flat)
        return self.fc(x)
