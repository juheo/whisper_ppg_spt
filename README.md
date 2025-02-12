## Python usage

get audio features from whisper encoder can be performed within Python: 

```python
import torch 
import librosa 
import julius 

from model import ModelDimensions, AudioEncoder
from utils import center, get_segments, log_mel_spectrogram, mel_filters

# load model
ckpt_file = "/nas/public/model/whisper/large-v2-encoder.pt"
checkpoint = torch.load(ckpt_file, map_location=device)
dims = ModelDimensions(**checkpoint["dims"])

whisper_encoder = AudioEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
whisper_encoder.load_state_dict(checkpoint["model_state_dict"])
whisper_encoder.to(device)
whisper_encoder.eval()

# load mel filter
filters = mel_filters(device, 80)

# test batch wave
audio = torch.randn(4, 128*512-1).to(device) # wave length should be (N*512-1) -> than ppg length is N

# get mel
wav_length = audio.shape[1]
center_point_idx = center(wav_length, win_length=2048, hop_length=256, sr1=44100, sr2=16000)
audio_16k = julius.resample_frac(audio, 44100, 16000)
segs = get_segments(audio_16k, center_point_idx, segment_length=400)
mel = log_mel_spectrogram(segs, filters) 

# get_ppg
with torch.no_grad():
    ppg = whisper_encoder(mel)
    print(ppg.shape) # [B, T, 1280]

```