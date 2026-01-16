import torch
from torch import nn
from torch.nn import functional as F

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


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
            self,
            resolutions,
            vocoder,
            factor_sc,
            factor_mag
    ):
        super(MultiResolutionSTFTLoss, self).__init__()

        self.resolutions = resolutions
        self.vocoder = vocoder
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.epsilon = torch.finfo(torch.float32).eps

        self.vocoder.eval()
        for p in self.vocoder.parameters():
            p.requires_grad = False

        self._windows = {}

    def _window(self, win_length, x: torch.Tensor):
        key = (win_length, x.device, x.dtype)
        if key not in self._windows:
            self._windows[key] = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        return self._windows[key]

    def spectral_convergence_loss(
            self,
            pred_mag: torch.Tensor,
            true_mag: torch.Tensor
    ):
        return torch.norm(true_mag - pred_mag, p="fro") / (torch.norm(true_mag, p="fro") + self.epsilon)

    def log_stft_magnitude_loss(
            self,
            pred_mag,
            true_mag
    ):
        return F.l1_loss(torch.log(self.epsilon + true_mag), torch.log(self.epsilon + pred_mag))

    def forward(
            self, batch_pred_mel: torch.Tensor,
            batch_true_mel: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor
    ) -> float:

        B, T, M = batch_pred_mel.shape
        # B
        mel_lens = (~tgt_key_padding_mask).sum(dim=1)

        sc_loss = torch.tensor(0.0, device=batch_pred_mel.device)
        mag_loss = torch.tensor(0.0, device=batch_pred_mel.device)
        valid_count = 0

        for i in range(B):
            Lm = int(mel_lens[i].item())
            if Lm <= 0:
                continue

            valid_count += 1

            # B, M, T
            pred_mel_i = batch_pred_mel[i:i + 1, :Lm, :].transpose(2, 1).float()
            true_mel_i = batch_true_mel[i:i + 1, :Lm, :].transpose(2, 1).float()


            # B. 1, T --> B, T
            pred_wav_i = self.vocoder(pred_mel_i).squeeze(1)
            with torch.no_grad():
                true_wav_i = self.vocoder(true_mel_i).squeeze(1)

            pred_wav_i = pred_wav_i.to(torch.float32)
            true_wav_i = true_wav_i.to(torch.float32)

            sample_sc_loss = 0
            sample_mag_loss = 0

            for (n_fft, win_length, hop_length) in self.resolutions:
                window = self._window(win_length, pred_wav_i)

                pred_stft = torch.stft(
                    pred_wav_i, n_fft, hop_length, win_length,
                    window=window, return_complex=True
                )

                true_stft = torch.stft(
                    true_wav_i, n_fft, hop_length, win_length,
                    window=window, return_complex=True
                )

                pred_mag = torch.abs(pred_stft)
                true_mag = torch.abs(true_stft)

                sample_sc_loss += self.spectral_convergence_loss(pred_mag, true_mag)
                sample_mag_loss += self.log_stft_magnitude_loss(pred_mag, true_mag)

            sc_loss += sample_sc_loss / len(self.resolutions)
            mag_loss += sample_mag_loss / len(self.resolutions)

        if valid_count > 0:
            sc_loss = sc_loss / valid_count
            mag_loss = mag_loss / valid_count

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class SynthiaLoss(nn.Module):
  def __init__(
          self,
          ga_sigma,
          ga_alpha,
          ga_n_layer,
          ga_n_head_start,
          reduction_factor,
          pos_weight,
          mrstft_loss = None,
          mrstft_weight = None
  ):
      super().__init__()

      self.guided_attention = GuidedAttentionLoss(
          ga_sigma,
          ga_alpha,
          ga_n_layer,
          ga_n_head_start,
          reduction_factor = reduction_factor
      )

      self.bce_criterion = nn.BCEWithLogitsLoss(
          pos_weight=pos_weight,
          reduction="none"
      )

      self.mrstft_loss = mrstft_loss
      self.mrstft_weight = 0.0 if mrstft_weight is None else mrstft_weight

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

      mel_base_loss   = self.calc_l1_(mel_base, mel_true, valid_mask_mse)
      mel_final_loss  = self.calc_l1_(mel_final, mel_true, valid_mask_mse)

      guided_attention_loss = self.guided_attention(
          cross_attention,
          tokens_lens,
          mels_lens
      )

      stop_loss = self.bce_criterion(stop_pred, stop_true)
      stop_loss = (stop_loss * valid_mask_bce).sum() / (valid_mask_bce.sum() + 1e-8)

      mrstft_loss = torch.tensor(0.0, device=mel_final.device)
      if self.mrstft_loss is not None:
        mrstft_loss = self.mrstft_loss(mel_final, mel_true, tgt_key_padding_mask)


      return (
          mel_base_loss,
          mel_final_loss,
          guided_attention_loss,
          stop_loss,
          mrstft_loss * self.mrstft_weight
      )

  @staticmethod
  def calc_l1_(mel_pred, mel_true, valid_mask):
      # Логика не учитывания паддинга
      # B, T, M
      mae             = (mel_pred - mel_true).abs()
      mel_loss        = (mae * valid_mask).sum() / (valid_mask.sum() * mel_pred.size(-1) + 1e-8)
      return mel_loss