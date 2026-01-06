import torch
from torch import nn

class GuidedAttentionLoss(nn.Module):
    def __init__(
            self,
            sigma,
            alpha,
            n_layer,
            n_head_start,
            reduction_factor,
            reset_always = True,
            device = "cuda"
    ):
        super().__init__()

        self.sigma              = sigma
        self.alpha              = alpha
        self.reset_always       = reset_always
        self.n_layer            = n_layer
        self.n_head_start       = n_head_start
        self.guided_attn_masks  = None
        self.masks              = None
        self.reduction_factor   = reduction_factor
        self.device = device


    def _reset_masks(self):
        self.guided_attn_masks  = None
        self.masks              = None

    def forward(
            self,
            cross_attn_list,  # B, heads, tgt_T, src_T
            input_lens,       # B,
            output_lens       # B
    ):


        # B, heads, tgt_T, src_T
        selected_layer = cross_attn_list[self.n_layer]
        # B, 2, tgt_T, src_T
        attn           = selected_layer[:, self.n_head_start:self.n_head_start + 2]

        if self.guided_attn_masks is None:
          # B, 1, tgt_T, src_T
          self.guided_attn_masks = self._make_guided_attention_masks(
              input_lens, output_lens
          ).unsqueeze(1)

        if self.masks is None:
          # B, 1, tgt_T, src_T
          self.masks = self._make_masks(input_lens, output_lens).unsqueeze(1)

        # B, 2, tgt_T, src_T
        self.masks = self.masks.expand(-1, attn.size(1), -1, -1)


        # B, 2, tgt_T, src_T
        losses  = self.guided_attn_masks * attn
        # float
        loss    = (losses * self.masks.float()).sum() / (self.masks.sum() + 1e-8)

        if self.reset_always:
          self._reset_masks()

        return loss * self.alpha

    def _make_guided_attention_masks(
            self,
            input_lens,
            output_lens
    ):

        if self.reduction_factor > 1:
            output_lens = (output_lens + self.reduction_factor - 1) // self.reduction_factor

        B               = len(input_lens)
        max_input_len   = int(input_lens.max().item())
        max_output_len  = int(output_lens.max().item())

        guided_attn_masks = torch.zeros((B, max_output_len, max_input_len), dtype=torch.float32, device=self.device)

        for idx, (input_len, output_len) in enumerate(zip(input_lens, output_lens)):
            input_len   = int(input_len.item())
            output_len  = int(output_len.item())
            guided_attn_masks[idx, :output_len, :input_len] = self._make_guided_attention_mask(
                input_len, output_len, self.sigma
            )

        return guided_attn_masks



    def _make_guided_attention_mask(
            self,
            input_len,
            output_len,
            sigma
    ):

        grid_x, grid_y = torch.meshgrid(
        torch.arange(output_len, dtype=torch.float32, device=self.device),
        torch.arange(input_len, dtype=torch.float32, device=self.device),
        indexing="ij"
        )

        # output_lens, input_lens
        return 1.0 - torch.exp(
            -((grid_y / input_len - grid_x / output_len) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(
            self,
            input_lens,
            output_lens
    ):
        if self.reduction_factor > 1:
            output_lens = (output_lens + self.reduction_factor - 1) // self.reduction_factor

        B               = len(input_lens)
        max_input_len   = int(input_lens.max().item())
        max_output_len  = int(output_lens.max().item())

        input_masks   = torch.zeros((B, max_input_len), dtype=torch.bool, device=self.device)
        output_masks  = torch.zeros((B, max_output_len), dtype=torch.bool, device=self.device)

        for idx, (input_len, output_len) in enumerate(zip(input_lens, output_lens)):
            input_len                       = int(input_len.item())
            output_len                      = int(output_len.item())
            input_masks[idx, :input_len]    = True
            output_masks[idx, :output_len]  = True

        return output_masks.unsqueeze(-1) & input_masks.unsqueeze(-2)


class SynthiaLoss(nn.Module):
  def __init__(
          self,
          ga_sigma,
          ga_alpha,
          ga_n_layer,
          ga_n_head_start,
          reduction_factor,
          pos_weight,
          mel_pad_value,
          silence_margin,
          silence_weight,
          high_freq_ratio  = None,
          high_freq_weight = None
  ):
      super().__init__()

      self.guided_attention = GuidedAttentionLoss(
          ga_sigma,
          ga_alpha,
          ga_n_layer,
          ga_n_head_start,
          reduction_factor = reduction_factor
      )

      self.bce_criterion= nn.BCEWithLogitsLoss(
          pos_weight=pos_weight,
          reduction="none"
      )

      # freq_weighted_l1
      self.high_freq_ratio  = high_freq_ratio
      self.high_freq_weight = high_freq_weight

      # silence_penalty
      self.mel_pad_value    = mel_pad_value
      self.silence_margin   = silence_margin
      self.silence_weight   = silence_weight

  def forward(
          self,
          mel_base,
          mel_final,
          mel_true,
          tgt_key_padding_mask,
          dec_tgt_padding_mask,
          cross_attention,
          tokens_lens,
          mels_lens,
          stop_pred,
          stop_true
  ):
      # B, T, M
      valid_mask_mse  = (~tgt_key_padding_mask).float().unsqueeze(-1)
      valid_mask_bce  = (~dec_tgt_padding_mask)

      if self.high_freq_ratio is not None and self.high_freq_weight is not None:
          mel_base_loss = self.freq_weighted_l1(
              mel_base,
              mel_true,
              valid_mask_mse,
              self.high_freq_ratio,
              self.high_freq_weight
          )
          mel_final_loss = self.freq_weighted_l1(
              mel_final,
              mel_true,
              valid_mask_mse,
              self.high_freq_ratio,
              self.high_freq_weight
          )
      else:
          mel_base_loss   = self.calc_l1_(mel_base, mel_true, valid_mask_mse)
          mel_final_loss  = self.calc_l1_(mel_final, mel_true, valid_mask_mse)

      silence_loss = self.silence_penalty(
          mel_final,
          mel_true,
          valid_mask_mse,
          self.mel_pad_value,
          self.silence_margin,
          self.silence_weight
      )

      guided_attention_loss = self.guided_attention(
          cross_attention,
          tokens_lens,
          mels_lens
      )

      stop_loss = self.bce_criterion(stop_pred, stop_true)
      stop_loss = (stop_loss * valid_mask_bce).sum() / (valid_mask_bce.sum() + 1e-8)


      return (
          mel_base_loss,
          mel_final_loss,
          silence_loss,
          guided_attention_loss,
          stop_loss
      )

  @staticmethod
  def calc_l1_(mel_pred, mel_true, valid_mask):
      # Логика не учитывания паддинга
      # B, T, M
      mae             = (mel_pred - mel_true).abs()
      mel_loss        = (mae * valid_mask).sum() / (valid_mask.sum() * mel_pred.size(-1) + 1e-8)
      return mel_loss

  @staticmethod
  def freq_weighted_l1(
          mel_pred,
          mel_true,
          valid_mask,
          high_freq_ratio,
          high_freq_weight
  ):
      B, T, M           = mel_pred.size()
      # 80 * (1.0 - 0.5) = 80 * 0.5 = 40
      split             = int(M * (1.0 - high_freq_ratio))
      mae               = (mel_pred - mel_true).abs()
      # B, T, :40
      low_part          = mae[..., :split]
      # B, T, 40:
      high_part         = mae[..., split:]

      # B, T, M
      vm                = valid_mask.expand(B, T, M)
      # B, T, :40
      vm_low            = vm[..., :split]
      # B, T, 40:
      vm_high           = vm[..., split:]

      low_loss          = (low_part * vm_low).sum() / (vm_low.sum() + 1e-8)
      high_loss         = (high_part * vm_high).sum() / (vm_high.sum() + 1e-8)

      return low_loss + high_freq_weight * high_loss

  @staticmethod
  def silence_penalty(
          mel_pred,
          mel_true,
          valid_mask,
          mel_pad_value,
          silence_margin,
          silence_weight
  ):
      threshold = mel_pad_value + silence_margin
      # B, T, 1
      silent_mask = (mel_true < threshold).all(dim=-1, keepdim=True)
      silent_mask = silent_mask & valid_mask.bool()

      if silent_mask.sum() == 0:
          return mel_pred.new_tensor(0.0)

      err = (mel_pred - mel_true).abs()
      pen = (err * silent_mask).sum() / (silent_mask.float().sum() * mel_pred.size(-1) + 1e-8)
      return pen * silence_weight
