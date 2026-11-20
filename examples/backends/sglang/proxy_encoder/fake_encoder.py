import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

D, D_LLM, HIDDEN = 1280, 3584, 5120
MEL, Q, P = 128, 256, 257  # P = Q + 1 keeps L = P + 1024 > Q for every item
PATCH, IN_CH = 14, 6  # Qwen patch size; 3 channels x temporal 2
TILE = 448
TILE_GRID = TILE // PATCH  # grid per item: [1, 32, 32]
TILE_PATCHES = TILE_GRID * TILE_GRID  # 1024
EXPECTED_PARAMS = 66_581_504


def ffn(d_in, d_out):
    return nn.Sequential(nn.Linear(d_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, d_out))


class MLP1Image(nn.Module):
    """[N, 1176] -> [1, N, D]; 1176 = IN_CH * PATCH * PATCH per row."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(IN_CH, D, PATCH, stride=PATCH)
        self.ffn = ffn(D, D)

    def forward(self, x):
        x = self.conv(x.reshape(-1, IN_CH, PATCH, PATCH)).mean(
            dim=(2, 3)
        )  # [N,D,1,1] -> [N,D]
        return self.ffn(x).unsqueeze(0)


class MLP2Audio(nn.Module):
    """[1, MEL, T] -> [1, T, D]. Not called in the Qwen2.5-VL path; weights kept only (part of the architecture)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(MEL, D, 3, padding=1)
        self.ffn = ffn(D, D)

    def forward(self, x):
        return self.ffn(self.conv(x).transpose(1, 2))


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(1, P, D) * 0.02)
        self.mlp1 = MLP1Image()
        self.mlp2 = MLP2Audio()
        self.mlp3 = ffn(D, D)
        self.mlp4 = ffn(D, D_LLM)

    @property
    def dtype(self):
        return self.prefix.dtype

    @property
    def device(self):
        return self.prefix.device

    def _tile(self, patches):
        """[1024, 1176] -> [Q, D_LLM]."""
        c = self.mlp3(
            torch.cat([self.prefix, self.mlp1(patches)], dim=1)
        )  # [1, L, D], L = P + 1024 > Q
        return self.mlp4(c[:, -Q:]).squeeze(0)

    @torch.no_grad()
    def forward_tiles(self, pixel_values, grid_thw):
        """pixel_values [1024*items, 1176], grid_thw [n_img, 3] with rows [1, 32, 32n] -> [Q*items, D_LLM]."""
        out, row = [], 0
        for t, h, w in grid_thw.tolist():
            assert t == 1 and h == TILE_GRID and w % TILE_GRID == 0, (t, h, w)
            n = TILE_PATCHES * (w // TILE_GRID)
            for seg in pixel_values[row : row + n].split(TILE_PATCHES):
                out.append(self._tile(seg))
            row += n
        assert row == pixel_values.shape[0], (row, pixel_values.shape)
        return torch.cat(out, dim=0)


def load_or_init(path):
    """Load the weights if present; otherwise build them (seed 0, bf16, CPU), check the param count, save, then load.

    The returned model is built under the caller's default dtype/device.
    """
    if not os.path.isfile(path):
        prev = torch.get_default_dtype()
        torch.manual_seed(0)
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cpu"):
                init = FakeEncoder()
        finally:
            torch.set_default_dtype(prev)
        total = sum(p.numel() for p in init.parameters())
        assert total == EXPECTED_PARAMS, total
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file(init.state_dict(), path)
    model = FakeEncoder()
    model.load_state_dict(load_file(path))
    return model
