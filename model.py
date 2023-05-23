import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        B, T, C = x.shape
        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        assert T <= self.positional_embedding.shape[0], "audio too long"
        x = (x + self.positional_embedding[:T,:]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

    def chunk(self, feat, window_size=1500, hop_size=750, max_batch=4):
        # feat: [C, T]
        feat_list = []
        original_shape = feat.shape[-1]
        if original_shape <= window_size:
            return [feat.unsqueeze(0)], original_shape//2
        else:
            N = int(np.ceil((original_shape - window_size) / (hop_size))) + 1
            L = hop_size * (N - 1) + window_size
            feat_pad = torch.cat(
                [feat, torch.zeros_like(feat[:, : L - original_shape])], dim=1
            )
            feat_pad_chunk = [
                feat_pad[:, i * hop_size : i * hop_size + window_size] for i in range(N)
            ]
            feat_batch = torch.stack(feat_pad_chunk, dim=0)
            feat_list = []
            for i in range((int)(np.ceil(feat_batch.size()[0] / max_batch))):
                feat_list.append(feat_batch[i * max_batch : (i + 1) * max_batch])
            return feat_list, original_shape//2

    def unchunk(self, feat_list, original_shape, window_size=750, hop_size=375):
        # feat_list : list of [B, C, T]
        residual = window_size - hop_size
        result = None
        B, C, _ = feat_list[0].size()
        mask = (torch.arange(residual) / residual).to(feat_list[0].device).unsqueeze(0)
        mask = torch.tile(mask, (C, 1))
        for feat in feat_list:
            for b in range(feat.size()[0]):
                if result is None:
                    result = feat[b]
                else:
                    result[:, -residual:] = (
                        result[:, -residual:] * (1 - mask) + feat[b][:, :residual] * mask
                    )
                    result = torch.cat([result, feat[b][:, residual:]], dim=-1)
        return result[:, :original_shape]    

    def inference(self, x):
        # x: [C, T]
        # output: [C', T]
        output_list = []
        x_list, original_length = self.chunk(x)
        for x_temp in x_list:
            output = self.forward(x_temp)
            output_list.append(output.permute(0,2,1))
        result = self.unchunk(output_list, original_length)
        return result 



if __name__ == '__main__':

    gpu = 0
    device = f"cuda:{gpu}"
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # load model
    ckpt_file = "/nas/public/model/whisper/large-v2-encoder.pt"
    checkpoint = torch.load(ckpt_file, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])

    model = AudioEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, dims.n_audio_layer)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cuda(gpu)
    model.eval()

    print(count_parameters(model))

    mel = torch.randn(80, 5000).cuda(gpu)

    import time 
    start = time.time()
    with torch.no_grad():
        for i in range(10):
            output = model.inference(mel)

    print(time.time()-start)

    print(mel.shape)
    print(output.shape)
