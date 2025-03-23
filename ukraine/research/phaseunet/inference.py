import torch
import torchaudio
from .model import DNSModel
from typing import Type, Optional
import warnings
from .utils import (
    complex_spectrogram_to_waveform,
    single_waveform_inference,
    HP
)


def denoise_audio(
        model_weights_path: str,
        audio_path: str,
        device: str,
        hp: Type[HP] = HP,
        save: bool = True,
        save_path: Optional[str] = None
) -> Optional[torch.Tensor]:
    """
    NOTE: input audio sample rate that is different from 16000
    will require resample that may slightly affect output quality.

    Denoises input audio file and optionally saves the denoised audio.

    Args:
      model_weights_path: path to model weights file.
      audio_path: path to noisy audio.
      hp: hyperparameter class.
      device: str: 'cuda' or 'cpu'.
      save: whether to save the output audio.
      save_path: path to save denoised audio if `save=True`.

    Returns:
      If `save` is False, returns the denoised waveform as torch.Tensor.
      If `save` is True, saves the audio and returns None.
        """
    model = DNSModel()
    model.load_state_dict(
        torch.load(model_weights_path,
                   weights_only=True, map_location=torch.device(device)
                   )
    )
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != hp.sample_rate:
        warnings.warn(
            f"Audio sample rate: {sample_rate} is different from one the model "
            f"was trained on: {hp.sample_rate}, doing resampling.",
            UserWarning
        )
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=hp.sample_rate
        )
    denoised_spectrogram = single_waveform_inference(
        waveform, model, hp, device
    ).transpose(3, 2)

    denoised_waveform = complex_spectrogram_to_waveform(
        denoised_spectrogram[:, 0, :, :].cpu(),
        denoised_spectrogram[:, 1, :, :].cpu(),
        hp
    )

    if save:
        if save_path is None:
            raise ValueError(
                "save_path must be provided for saving denoised audio."
            )
        torchaudio.save(
            save_path,
            denoised_waveform,
            hp.sample_rate
        )
        return None
    else:
        return denoised_waveform
