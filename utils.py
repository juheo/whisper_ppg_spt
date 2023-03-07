import numpy as np 
import torch 
import torch.nn.functional as F


def center(wav_length, win_length=2048, hop_length=512, sr1=44100, sr2=16000):
    center_point = []
    wav_length = (win_length // 2) + wav_length + (win_length // 2)
    for i in range((wav_length - win_length) // hop_length + 1):
        center_point.append(i * hop_length)
    return np.asarray(np.asarray(center_point) * sr2 / sr1, dtype=int)

def get_segments(wav, center_point_idx, segment_length=400):
    # wav shape: B X T
    wav_length = wav.shape[1]
    segments = []
    for i in center_point_idx:
        l = segment_length // 2
        start = i - l
        end = i + l
        if start < 0:
            seg = wav[:, 0:end]
            seg = F.pad(seg, (np.abs(start), 0), mode="constant")
        elif end > wav_length:
            seg = wav[:, start:]
            seg = F.pad(seg, (0, end - wav_length), mode="constant")
        else:
            seg = wav[:, start:end]

        assert seg.shape[1] == segment_length

        segments.append(seg)
    return torch.stack(segments, dim=1)

def mel_filters(device, n_mels) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:
        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("/nas/public/model/whisper/mel_filters.npz") as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(segs, filters):

    N_FFT = 400
    N_MELS = 80
    HOP_LENGTH = 400

    """
    Compute the log-Mel spectrogram of
    Parameters
    ----------
    segs: torch.Tensor, shape = (B, T, 400)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz
    n_mels: int
        The number of Mel-frequency filters, only 80 is supported
    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    # segs [B,T,400] -> [B*T, 400]
    B, T, _ = segs.shape
    segs = segs.view(-1, segs.shape[-1])
    window = torch.hann_window(N_FFT).to(segs[0].device)
    stft = torch.stft(segs[:,:-1], N_FFT, HOP_LENGTH, window=window, return_complex=True).squeeze(-1)
    # stft [B*T, n_fft] -> [B, T, n_fft]
    stft = stft.view(B, T, stft.shape[-1])
    stft = stft.permute(0,2,1)
    magnitudes = stft.abs() ** 2
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec