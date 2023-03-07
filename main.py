import torch 
import librosa 
import julius 

from whisper.model import Whisper, ModelDimensions
from utils import center, get_segments, log_mel_spectrogram, mel_filters

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


whisper = load_model("/nas/public/model/whisper/large-v2.pt")
device = whisper.device
filters = mel_filters(whisper.device, 80)

whisper.eval()


with torch.no_grad():
    # test audio
    audio_path = "/nas/public/singing/english/sr44100/wav-only/6to8/F005_Jodi Benson_Part of Your World_가성/F005_Jodi Benson_Part of Your World_가성_1.wav"
    audio, sr = librosa.load(audio_path, sr=None)
    wav_length = len(audio)
    center_point_idx = center(wav_length, win_length=2048, hop_length=256, sr1=44100, sr2=16000)
    audio_16k = julius.resample_frac(torch.from_numpy(audio), 44100, 16000).unsqueeze(0).to(device)
    segs = get_segments(audio_16k, center_point_idx, segment_length=400)
    mel = log_mel_spectrogram(segs, filters)
    # get_ppg
    ppg = whisper.encoder(mel) # [B, T, 1280]
    print(ppg.shape)

    # plot
    plt.figure(figsize=(16,9))
    plt.subplot(2,1,1)
    librosa.display.specshow(mel.squeeze().detach().cpu().numpy())
    plt.subplot(2,1,2)
    librosa.display.specshow(ppg.squeeze().detach().cpu().numpy().T)
    plt.tight_layout()
    plt.savefig("mel_ppg.png")
    plt.close()

    # test batch wave
    audio = torch.randn(4, 128*512-1).to(device) # wave length should be (N*512-1) -> than ppg length is N
    wav_length = audio.shape[1]
    center_point_idx = center(wav_length, win_length=2048, hop_length=256, sr1=44100, sr2=16000)
    audio_16k = julius.resample_frac(audio, 44100, 16000)
    segs = get_segments(audio_16k, center_point_idx, segment_length=400)
    mel = log_mel_spectrogram(segs, filters)
    # get_ppg
    ppg = whisper.encoder(mel) # [B, T, 1280]
    print(ppg.shape)

