import torch
from torch import nn
from ukraine.research.transformer.layers import PositionalEncoding
from ukraine.research.transformer.transformer import EncoderLayer, DecoderLayer


class EncoderPreNet(nn.Module):
  def __init__(self, d_model, dropout_rate):
    super().__init__()

    self.net = nn.Sequential(
        self.conv_block(d_model, d_model, dropout_rate),
        self.conv_block(d_model, d_model, dropout_rate),
        self.conv_block(d_model, d_model, dropout_rate)
    )

    self.proj = nn.Linear(d_model, d_model)

  def forward(self, x):
    x = x.transpose(1, 2)
    x = self.net(x)
    x = x.transpose(1, 2)
    x = self.proj(x)
    return x


  @staticmethod
  def conv_block(in_c, out_c, dropout_rate):
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=5, padding=2),
        nn.GroupNorm(num_groups=8, num_channels=out_c),
        nn.ReLU(),
        nn.Dropout(dropout_rate)
        )


class DecoderPreNet(nn.Module):
  def __init__(self, d_model, reduction_factor, dropout_rate):
    super().__init__()

    self.net = nn.Sequential(
        nn.Linear(80 * reduction_factor, d_model),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Dropout(dropout_rate)
    )

  def forward(self, tgt):
    return self.net(tgt)


class Encoder(nn.Module):
  def __init__(
      self,
      num_encoder_layers,
      d_model,
      num_heads,
      input_vocab_size,
      dropout_rate,
      encoder_dropout_rate,
      ff_factory,
      norm_factory,
      pad_token_id,
      use_flash
      ):
    super(Encoder, self).__init__()

    self.spe              = PositionalEncoding(d_model, scaled=True)
    self.encoder_pre_net  = EncoderPreNet(d_model, encoder_dropout_rate)
    self.embedding        = nn.Embedding(input_vocab_size, d_model, padding_idx=pad_token_id)
    self.dropout          = nn.Dropout(dropout_rate)
    self.d_model          = d_model

    self.encoder_layers   = nn.ModuleList(
        [
            EncoderLayer(
                d_model, num_heads,
                dropout_rate, ff_factory,
                norm_factory, use_flash=use_flash
            )
            for _ in range(num_encoder_layers)
        ]
    )
    self.final_layer_norm = norm_factory()

  def forward(
      self,
      x,
      src_mask = None,
      src_key_padding_mask = None
      ):

    x = self.embedding(x.to(torch.long))
    x = self.encoder_pre_net(x)
    x = self.dropout(self.spe(x))

    for layer in self.encoder_layers:
      x = layer(x, src_mask, src_key_padding_mask)
    x = self.final_layer_norm(x)

    return x


class Decoder(nn.Module):
  def __init__(
      self,
      num_decoder_layers,
      d_model,
      num_heads,
      dropout_rate,
      decoder_dropout_rate,
      ff_factory,
      norm_factory,
      reduction_factor,
      use_flash,
      use_cross_attn
      ):
    super().__init__()

    self.decoder_prenet = DecoderPreNet(d_model, reduction_factor, decoder_dropout_rate)
    self.spe = PositionalEncoding(d_model, scaled=True)
    self.positional_dropout = nn.Dropout(dropout_rate)

    self.decoder_layers = nn.ModuleList(
        [
            DecoderLayer(
                d_model, num_heads,
                dropout_rate, ff_factory,
                norm_factory, use_flash, use_cross_attn, cross_fusion=True
            )
            for _ in range(num_decoder_layers)
        ]
    )

    self.final_norm   = norm_factory()
    self.d_model      = d_model

  def forward(
      self,
      tgt,
      memory,
      tgt_mask                = None,
      memory_mask             = None,
      tgt_key_padding_mask    = None,
      memory_key_padding_mask = None
      ):

    x = self.decoder_prenet(tgt)
    x = self.positional_dropout(self.spe(x))

    cross_attns = []

    for layer in self.decoder_layers:
      x, _, cross_attn = layer(
          x,
          memory,
          tgt_mask                  = tgt_mask,
          memory_mask               = memory_mask,
          tgt_key_padding_mask      = tgt_key_padding_mask,
          memory_key_padding_mask   = memory_key_padding_mask
        )
      cross_attns.append(cross_attn)

    x = self.final_norm(x)

    return x, cross_attns


class PostNet(nn.Module):
  def __init__(self, d_model, reduction_factor, dropout_rate):
    super().__init__()

    ch_ = [80 * reduction_factor, d_model, d_model * 2, d_model * 2, d_model]

    self.net = nn.ModuleList()

    for i in range(len(ch_) - 1):
      self.net.append(
          nn.Sequential(
              nn.Conv1d(ch_[i], ch_[i+1], kernel_size=5, padding=2),
              nn.GroupNorm(num_groups=8, num_channels=ch_[i+1]),
              nn.Tanh(),
              nn.Dropout(dropout_rate)
              )
          )

    self.final_conv = nn.Conv1d(d_model, 80 * reduction_factor, kernel_size=5, padding=2)

  def forward(self, x):
    # B, T, M -> B, M, T
    x   = x.transpose(2, 1)

    for layer in self.net:
      x = layer(x)

    x   = self.final_conv(x)
    # B, M, T -> B, T, M
    x   = x.transpose(2, 1)

    return x
