"""Audio segmentation: log-mel of the whole clip, cut into time segments. Standalone, not wired into Dynamo.

Needs librosa (imported lazily).
"""

import io
import logging
import os
from typing import List, Union

import msgspec
import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
# Moving-average width (80 ms) used to find a quiet cut point.
_QUIET_SMOOTH_FRAMES = 8


class AudioChunking(msgspec.Struct, frozen=True, kw_only=True):
    """Long-audio chunking configuration."""

    enabled: bool
    chunk_frames: int
    max_chunks: int
    overlap_frames: int = 0
    min_tail_frames: int = 100
    search_ratio: float = 0.94

    @classmethod
    def from_env(cls) -> "AudioChunking":
        enabled = os.environ.get(
            "FAKE_ENABLE_AUDIO_CHUNKING", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        chunk = max(int(os.environ.get("FAKE_AUDIO_CHUNK_FRAMES", "2704")), 1)
        max_chunks = max(int(os.environ.get("FAKE_MAX_AUDIO_CHUNKS", "45")), 1)
        return cls(enabled=enabled, chunk_frames=chunk, max_chunks=max_chunks)


def _audio_source(payload: Union[str, bytes]) -> Union[str, io.BytesIO]:
    """Local path or bytes only."""
    if isinstance(payload, (bytes, bytearray)):
        return io.BytesIO(bytes(payload))
    if not isinstance(payload, str):
        raise ValueError(f"unsupported audio payload type: {type(payload)}")
    return payload


def load_waveform(payload: Union[str, bytes]) -> np.ndarray:
    """Mono float32 waveform resampled to 16 kHz."""
    import librosa

    waveform, _ = librosa.load(_audio_source(payload), sr=SAMPLING_RATE)
    if waveform.size == 0:
        # Fail early: power_to_db would raise a cryptic error on an empty waveform.
        desc = payload if isinstance(payload, str) else f"{len(payload)} bytes"
        raise ValueError(f"audio decodes to an empty waveform: {desc}")
    return waveform


def log_mel(waveform: np.ndarray, *, n_mels: int) -> torch.Tensor:
    """[n_mels, frames] ``power_to_db(mel, ref=max)`` of the whole clip."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
    )
    return torch.from_numpy(librosa.power_to_db(mel, ref=np.max)).float()


def _find_quiet_split(mel: torch.Tensor, lo: int, hi: int) -> int:
    """Quietest smoothed frame in [lo, hi); hi if the window is too narrow."""
    smooth = _QUIET_SMOOTH_FRAMES
    lo = max(int(lo), 0)
    hi = min(int(hi), int(mel.shape[1]))
    if hi - lo < smooth:
        return hi
    energy = mel[:, lo:hi].mean(dim=0)
    kernel = torch.ones(1, 1, smooth, dtype=energy.dtype) / smooth
    smoothed = torch.nn.functional.conv1d(energy.view(1, 1, -1), kernel).view(-1)
    return lo + int(torch.argmin(smoothed)) + smooth // 2


def split_mel(
    mel: torch.Tensor, *, chunking: AudioChunking, max_time_steps: int
) -> List[torch.Tensor]:
    """Cut the log-mel into time segments.

    Chunking off: a single mel[:, :max_time_steps]. On: each segment ends at the quietest frame in the last
    (1 - search_ratio) of its chunk_frames window; at most max_chunks segments; a tail shorter than
    min_tail_frames merges into the previous one.
    """
    total = int(mel.shape[1])
    if not chunking.enabled:
        return [mel[:, :max_time_steps]]

    chunk = chunking.chunk_frames
    spans = []
    start = 0
    while start < total and len(spans) < chunking.max_chunks:
        if start + chunk >= total:
            spans.append((start, total))
            break
        end = max(
            _find_quiet_split(
                mel, start + int(chunk * chunking.search_ratio), start + chunk
            ),
            start + 1,
        )
        spans.append((start, end))
        start = max(end - chunking.overlap_frames, start + 1)

    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < chunking.min_tail_frames:
        tail_end = spans.pop()[1]
        spans[-1] = (spans[-1][0], tail_end)

    dropped = total - spans[-1][1]
    if dropped > 0:
        logger.debug(
            "audio: dropped %d tail frames (%.1fs) after %d chunks; raise FAKE_MAX_AUDIO_CHUNKS to cover more",
            dropped,
            dropped * HOP_LENGTH / SAMPLING_RATE,
            len(spans),
        )
    return [mel[:, s:e] for s, e in spans]


def audio_segments(
    payload: Union[str, bytes],
    *,
    n_mels: int,
    max_time_steps: int,
    chunking: AudioChunking,
) -> List[torch.Tensor]:
    """One request audio -> its [n_mels, frames] mel segments, in time order."""
    mel = log_mel(load_waveform(payload), n_mels=n_mels)
    return split_mel(mel, chunking=chunking, max_time_steps=max_time_steps)


if __name__ == "__main__":
    segments = audio_segments(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "source", "1.wav"),
        n_mels=128,
        max_time_steps=5408,
        chunking=AudioChunking.from_env(),
    )
    print(f"segments={len(segments)}")
    for i, seg in enumerate(segments):
        print(f"  [{i}] {list(seg.shape)}")
