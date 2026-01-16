import torch

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(
            self,
            resolutions,
            vocoder,
            factor_sc,
            factor_mag,
            max_items,
            crop_frames=128
    ):
        super(MultiResolutionSTFTLoss, self).__init__()

        self.resolutions = resolutions
        self.vocoder = vocoder
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.epsilon = torch.finfo(torch.float32).eps

        self.max_items = max_items
        self.crop_frames = crop_frames

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

        K = min(self.max_items, B)
        idx = torch.randperm(B, device=batch_pred_mel.device)[:K]


        sc_loss = torch.tensor(0.0, device=batch_pred_mel.device)
        mag_loss = torch.tensor(0.0, device=batch_pred_mel.device)
        valid_count = 0

        for i in idx.tolist():
            L = int(mel_lens[i].item())
            if L <= 0:
                continue

            Lc = min(L, self.crop_frames)
            start = 0 if Lc == L else int(torch.randint(0, L - Lc+1, size=(1,), device=batch_pred_mel.device).item())

            # B, M, T
            pred_mel_i = batch_pred_mel[i:i + 1, start:start+Lc, :].transpose(2, 1).contiguous()
            true_mel_i = batch_true_mel[i:i + 1, start:start+Lc, :].transpose(2, 1).contiguous()


            # B. 1, T --> B, T
            pred_wav_i = self.vocoder(pred_mel_i).squeeze(1)
            with torch.no_grad():
                true_wav_i = self.vocoder(true_mel_i).squeeze(1)

            pred_wav_i = pred_wav_i.to(torch.float32)
            true_wav_i = true_wav_i.to(torch.float32)

            sample_sc_loss = 0
            sample_mag_loss = 0
            res_used = 0

            for (n_fft, win_length, hop_length) in self.resolutions:
                if pred_wav_i.size(-1) < n_fft:
                    continue

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
                res_used += 1

            if res_used == 0:
                continue

            sc_loss += sample_sc_loss / res_used
            mag_loss += sample_mag_loss / res_used
            valid_count += 1

        if valid_count > 0:
            sc_loss = sc_loss / valid_count
            mag_loss = mag_loss / valid_count

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss