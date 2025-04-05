import torch
from typing import Type
import urllib.request
import os


class HP:
    """
  sample_rate: dataset sample_rate.
  n_fft: n_fft parameters for torch.stft() function.
  win_length: win_length parameter for torch.stft() function.
  hop_length: hop_length parameter for torch.stft() function.
  """
    sample_rate = 16000
    n_fft = 382
    win_length = 256
    hop_length = 160


def transform_to_complex_spectrogram(
        wf: torch.Tensor, hp: Type[HP]
) -> torch.Tensor:
    """
  Transforms waveform to complex spectrogram using Short-time Fourier transform.
  torch.stft() originally returns spectrogram of shape: (1, n_mels, timeframes)
  that contains complex values of amplitude and phase. To make this data
  compatible with NN, we split the original complex spectrogram
  into real (amplitude) and imaginary (phase) 'channels' to obtain real numbers.
  Then we concatenate them to obtain phase-aware spectrogram and transpose it.

  Args:
    wf: waveform of shape: (1, time).
    hp: hyperparameter class.

  Returns: spectrogram of shape: (2, timeframes, n_mels) where 2 represents
  real and imaginary parts.

  """
    stft = torch.stft(
        input=wf,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window=torch.hann_window(hp.win_length),
        return_complex=True
    )
    real_part = stft.real
    image_part = stft.imag

    spec = torch.stack([real_part, image_part], dim=0).squeeze(1)
    spec = torch.transpose(spec, 2, 1)
    return spec


def complex_spectrogram_to_waveform(
        real: torch.Tensor, imag: torch.Tensor, hp: Type[HP]
) -> torch.Tensor:
    """
  Transforms complex spectrogram to waveform using
  Inverse Short-time Fourier transform.
  The function accepts both 3D tensors of real and imaginary data
  ex. [1, n_mels, timeframes] and 2D tensors [n_mels, timeframes].
  We then form a complex spectrogram and run through
  Inverse Short-time Fourier transform to obtain waveform.

  Args:
    real: amplitude part of spectrogram of shape: (n_mels, timeframes)
        or (1, n_mels, timeframes).
    imag: phase part of spectrogram of shape: (n_mels, timeframes)
        or (1, n_mels, timeframes).
    hp: hyperparameter class.

  Returns: waveform of shape: (time) if amplitude and phase are 2D tensors
  and (1, time) if amplitude and phase are 3D tensors.

  """
    complex_spectrogram = torch.complex(real, imag)
    istft = torch.istft(
        complex_spectrogram,
        n_fft=hp.n_fft,
        hop_length=hp.hop_length,
        win_length=hp.win_length,
        window=torch.hann_window(hp.win_length)
    )
    return istft


def single_waveform_inference(
        noisy_wf: torch.Tensor,
        denoise_model: torch.nn.Module, hp: Type[HP],
        device: str
) -> torch.Tensor:
    """
    NOTE: This function does not work with batches.

    Transforms noisy waveform into complex spectrogram
    and clean noisy complex spectrogram via pre-trained model.

    Args:
      noisy_wf: waveform of shape: (1, timeframe) or (timeframe).
      denoise_model: PyTorch model instance.
      hp: hyperparameter class.
      device: str: 'cuda' or 'cpu'.

    Returns: complex spectrogram of shape: (1, 2, timeframes, n_mels),
    where '2' represents real and imaginary parts.
    """
    denoise_model.eval()
    with torch.no_grad():
        spectrogram = transform_to_complex_spectrogram(noisy_wf, hp)
        pred_spectrogram = denoise_model(spectrogram.unsqueeze(0).to(device))
    return pred_spectrogram


def download_phaseunet_weights(save_dir: str) -> None:
    """
    Downloads the "Phase-U-Net" model weights to the specified directory. If the
    weights already exist at the target location, the download is skipped and
    appropriate message is displayed.

    :param save_dir: Directory where the weights should be saved.
    :type save_dir: str
    :return: None    :rtype: None
    """
    url = "https://github.com/anarlavrenov/ukraine/releases/download/v0.0.7/model_state_dict_epoch_29_pesq_2.946.pth"
    save_dir = os.path.expanduser(save_dir)
    save_path = os.path.join(
        save_dir,
        "phase_u_net_weights.pth"
    )
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_path):
        print(f"Downloading phaseunet weights to {save_path}")
        urllib.request.urlretrieve(url, save_path)
    else:
        print(f"Weights already exists at {save_path}")

    return None
