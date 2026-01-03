import torch
from torch import nn
from .layers import *


class Synthia(nn.Module):
  def __init__(
          self,
          num_encoder_layers,
          num_decoder_layers,
          d_model,
          num_heads,
          input_vocab_size,
          trf_dropout_rate,
          encoder_dropout_rate,
          decoder_dropout_rate,
          postnet_dropout_rate,
          ff_factory,
          norm_factory,
          pad_token_id,
          use_flash,
          reduction_factor,
          use_cross_attn = True
  ):
      super().__init__()

      # B, T, M -> B, T//r, M * r.
      # В DecoderPrenet заходит тензор  : B, T//r, M * r    -> на выходе B, T//r, d_model.
      # В Decoder заходит тензор        : B, T//r, M * r    -> на выходе B, T//r, d_model.
      # В output_linear заходит тензор  : B, T//r, d_model  -> на выходе B, T//r, M * r.
      # В PostNet заходит тензор        : B, T//r, M * r    -> на выходе B, T//r, M * r.
      self.encoder = Encoder(
          num_encoder_layers,
          d_model,
          num_heads,
          input_vocab_size,
          trf_dropout_rate,
          encoder_dropout_rate,
          ff_factory,
          norm_factory,
          pad_token_id,
          use_flash
      )

      self.decoder = Decoder(
          num_decoder_layers,
          d_model,
          num_heads,
          trf_dropout_rate,
          decoder_dropout_rate,
          ff_factory,
          norm_factory,
          reduction_factor,
          use_flash,
          use_cross_attn
      )

      self.postnet = PostNet(
          d_model,
          reduction_factor,
          postnet_dropout_rate
      )

      self.reduction_factor = reduction_factor
      self.n_mels           = 80
      self.output_linear    = nn.Linear(d_model, 80 * self.reduction_factor)
      self.stop_linear      = nn.Linear(d_model, 1)


  def forward(
          self,
          src,
          tgt,
          src_key_padding_mask,
          tgt_mask,
          tgt_key_padding_mask,
          memory_key_padding_mask
  ):

      memory                        = self.encoder(
          src,
          src_key_padding_mask      = src_key_padding_mask
      )

      decoder_out, cross_attention  = self.decoder(
          tgt,
          memory,
          tgt_mask                  = tgt_mask,
          tgt_key_padding_mask      = tgt_key_padding_mask,
          memory_key_padding_mask   = memory_key_padding_mask
      )

      stop_out                      = self.stop_linear(decoder_out).squeeze(-1)
      mel_base                      = self.output_linear(decoder_out)
      mel_delta                     = self.postnet(mel_base)
      mel_final                     = mel_base + mel_delta

      return mel_base, mel_final, cross_attention, stop_out
