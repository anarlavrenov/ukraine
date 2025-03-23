import torch
from typing import Tuple, Optional


class DNSModel(torch.nn.Module):
    """
  Deep Phase-Aware Noise Suppression model.
  The model has a UNet-like architecture: contracted part, bottleneck and
  expanding part with skip connections.
  Contracted part consists of three Convolutional blocks (conv_block).
  Each block consists of two Conv2d layers, two Batch Normalization layers
  and two ReLU activations.
  After each Convolutional block we use MaxPool2d
  to decrease height and width of tensor.
  Bottleneck is a Convolution block (conv_block) with 256 input channels
  and 512 output channels, two Batch Normalization layers
  and two ReLU activations.
  Expanding part consists of three Transposed Convolutional blocks
  (conv_transpose_block). In difference with Convolutional blocks,
  Transposed Convolutional blocks contain only one Conv2dTranspose layer
  with Batch Normalization.

  Contracted part starts with tensor of shape: (batch_size, 2, 400, 192)
  and ends with tensor of shape: (batch_size, 256, 50, 24).
  Middle part produces tensor of shape: (batch_size, 512, 50, 24)
  Expanding part starts with tensor of shape: (batch_size, 256, 100, 48) and
  after expanding blocks (including skip connections) ends with tensor
  of shape (batch_size, 64, 400, 192).

  The output layer is a Conv2d layer with 2 output channels
  as we work with complex (phase-aware) spectrograms.
  The raw model output is a tensor of shape (batch_size, 2, 400, 192)
  where '2' represents real and imaginary parts.


  Attributes:
    conv1 (torch.nn.Sequential): First convolutional block with 2 input channels,
        64 output channels, two Conv2d layers, two BatchNorm2d layers,
        and ReLU activation.
    pool1 (torch.nn.MaxPool2d): Max pooling layer after `conv1`
        to downsample the spatial dimensions.
    conv2 (torch.nn.Sequential): Second convolutional block with 64 input channels,
        128 output channels, two Conv2d layers, two BatchNorm2d layers, and ReLU activation.
    pool2 (torch.nn.MaxPool2d): Max pooling layer after `conv2`
        to downsample the spatial dimensions.
    conv3 (torch.nn.Sequential): Third convolutional block with 128 input channels,
        256 output channels, two Conv2d layers, two BatchNorm2d layers, and ReLU activation.
    pool3: Max pooling layer after `conv3`
        to downsample the spatial dimensions.
    middle: (torch.nn.Sequential): Bottleneck convolutional block
        with 256 input channels, 512 output channels, two Conv2d layers,
        two BatchNorm2d layers, and ReLU activation.
    conv_tp1: (torch.nn.Sequential): First transposed convolutional block
        with 512 input channels, 256 output channels, one ConvTranspose2d layer,
        and BatchNorm2d layer.
    concat_conv1: (torch.nn.Sequential): Convolutional block that processes
      concatenated features from `conv_tp1` and `conv3` with 512 input channels
      and 256 output channels.
    conv_tp2: (torch.nn.Sequential): Second transposed convolutional block with
      256 input channels, 128 output channels, one ConvTranspose2d layer,
      and BatchNorm2d layer.
    concat_conv2: (torch.nn.Sequential): Convolutional block that processes
      concatenated features from `conv_tp2` and `conv2` with 256 input channels
      and 128 output channels.
    conv_tp3: conv_tp3 (torch.nn.Sequential): Third transposed convolutional
      block with 128 input channels, 64 output channels, one ConvTranspose2d layer, and BatchNorm2d layer.
    concat_conv3: (torch.nn.Sequential): Convolutional block that processes concatenated features
        from `conv_tp3` and `conv1` with 128 input channels and 64 output channels.
    output_layer: (torch.nn.Conv2d): Final convolutional layer with 64 input channels
        and 2 output channels (real and imaginary parts of the spectrogram).

    Weights initialization for all Conv2d layers is applied
    via 'weights_init' method  inherited from DCGAN approach.
  """

    def __init__(self):
        super(DNSModel, self).__init__()

        self.conv1 = self.conv_block(
            2, 64, 3, 1, 1, True
        )
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=2
        )
        self.conv2 = self.conv_block(
            64, 128, 3, 1, 1, True
        )
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=2
        )
        self.conv3 = self.conv_block(
            128, 256, 3, 1, 1, True
        )
        self.pool3 = torch.nn.MaxPool2d(
            kernel_size=2
        )
        self.middle = self.conv_block(
            256, 512, 3, 1, 1, True
        )
        self.conv_tp1 = self.conv_transpose_block(
            512, 256, 3, 2, 1, 1, True
        )
        self.concat_conv1 = self.conv_block(
            512, 256, 3, 1, 1, True
        )
        self.conv_tp2 = self.conv_transpose_block(
            256, 128, 3, 2, 1, 1, True
        )
        self.concat_conv2 = self.conv_block(
            256, 128, 3, 1, 1, True
        )
        self.conv_tp3 = self.conv_transpose_block(
            128, 64, 4, 2, 1, 0, True
        )
        self.concat_conv3 = self.conv_block(
            128, 64, 3, 1, 1, True
        )
        self.output_layer = torch.nn.Conv2d(
            64, 2, 1, 1, 0, bias=False
        )

        self.weights_init()

    def forward(
            self, x: torch.Tensor, spec_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
    NOTE: 'align_tensors method' is applied to prevent dimensions mismatch
    on timeframe and mels axis.

    Args:
      x: tensor of shape (batch_size, 2, timeframes, n_mels).
      spec_lengths: tensor of shape: (batch_size).

    Returns: tensor of shape (batch_size, 2, timeframes, n_mels)
    where '2' represents real and imaginary parts.

    """
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        middle = self.middle(x6)

        x = self.conv_tp1(middle)
        x, x5 = self.align_tensors(x, x5)
        x = self.concat_conv1(torch.cat([x, x5], dim=1))

        x = self.conv_tp2(x)
        x, x3 = self.align_tensors(x, x3)
        x = self.concat_conv2(torch.cat([x, x3], dim=1))

        x = self.conv_tp3(x)
        x, x1 = self.align_tensors(x, x1)
        x = self.concat_conv3(torch.cat([x, x1], dim=1))

        out = self.output_layer(x)

        if spec_lengths is not None:
            return self.parse_output(out, spec_lengths)
        else:
            return out

    @staticmethod
    def conv_block(
            in_channels: int, out_channels: int,
            kernel_size: int, stride: int,
            padding: int, activation: Optional[bool] = False
    ) -> torch.nn.Sequential:
        """
    Returns: torch.nn.Sequential: the convolutional block.

    """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels,
                kernel_size, stride,
                padding, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels,
                kernel_size, stride,
                padding, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        )
        if activation is not False:
            block.append(torch.nn.ReLU())
        return block

    @staticmethod
    def conv_transpose_block(
            in_channels: int, out_channels: int,
            kernel_size: int, stride: int,
            padding: int, output_padding: int,
            activation: Optional[bool] = False
    ) -> torch.nn.Sequential:
        """

    Returns: torch.nn.Sequential: the transposed convolutional block.

    """
        block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size, stride,
                padding, output_padding, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels)
        )
        if activation is not False:
            block.append(torch.nn.ReLU())
        return block

    @staticmethod
    def align_tensors(
            x: torch.Tensor, xn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
    Aligns two tensors on height and width axis for subsequent their concatenation.
    The potential tensor mismatch can happen due to unpaired numbers of height
    and width after any of convolutional blocks.

    Args:
      x: tensor of shape: (batch_size, channels, height, width)
      xn: tensor of shape: (batch_size, channels, height, width)

    Returns: aligned tensors with shape (batch_size, channels, height, width)

    """
        if x.size(2) != xn.size(2):
            xn = xn[:, :, :x.size(2), :]
        if x.size(3) != xn.size(3):
            xn = xn[:, :, :, :x.size(3)]
        return x, xn

    @staticmethod
    def parse_output(
            raw_model_output: torch.Tensor,
            spec_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
    Fill out the timeframe axis with zeros using 'spec_lengths' parameter -
    original timeframe lengths before padding. This is done to ensure that
    the loss function does not account padding values.

    Args:
      raw_model_output: tensor of shape: (batch_size, 2, timeframes, n_mels)
      spec_lengths: tensor of shape: (batch_size)

    Returns:
        tensor of shape: (batch_size, 2, timeframes, n_mels)

    """
        batch_size, channels, max_timeframes, n_mels = raw_model_output.shape
        ids = torch.arange(0, max_timeframes).unsqueeze(0). \
            expand(batch_size, -1).to(raw_model_output.device)
        padding_mask = ids >= spec_lengths.unsqueeze(1). \
            expand(batch_size, max_timeframes).to(raw_model_output.device)

        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-1)
        padding_mask = padding_mask.expand(-1, 2, -1, n_mels)
        parsed_output = raw_model_output.masked_fill(padding_mask, 0.0)
        return parsed_output

    def weights_init(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
