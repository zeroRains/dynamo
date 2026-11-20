import functools
import importlib.util
import os

import torch
from fake_encoder import TILE, TILE_GRID, load_or_init
from tiling import TilingConfig, TilingImageProcessor

WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "fake_encoder.safetensors"
)


def _make_visual(*_args, **_kwargs):
    """Replace Qwen2_5_VisionTransformer: ignore its args and build FakeEncoder (weights generated if missing)."""
    model = load_or_init(WEIGHTS)
    p = next(model.parameters())
    print(f"[fake_model] visual dtype={p.dtype} device={p.device}", flush=True)
    return model


def _get_image_feature(self, items):
    pixel_values = torch.cat([item.feature for item in items], dim=0)
    grid = torch.cat([item.image_grid_thw for item in items], dim=0)
    rows = grid.tolist()
    print(
        f"[fake_model] grid={rows} items={[w // TILE_GRID for _, _, w in rows]} "
        f"rows={[t * h * w for t, h, w in rows]} total={pixel_values.shape[0]}",
        flush=True,
    )
    return self.visual.forward_tiles(
        pixel_values.to(self.visual.device, self.visual.dtype), grid
    )


def _get_video_feature(self, items):
    raise NotImplementedError("fake encoder: image only")


def apply():
    tiling_cfg = TilingConfig.from_env()

    import sglang
    from sglang.srt.disaggregation.encoder import preprocessor
    from sglang.srt.models import qwen2_5_vl

    qwen2_5_vl.Qwen2_5_VisionTransformer = _make_visual
    cls = qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
    cls.get_image_feature = _get_image_feature
    cls.get_video_feature = _get_video_feature

    orig_init = preprocessor.EncoderPreprocessor.__init__

    @functools.wraps(orig_init)
    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if self.image_processor is not None:
            self.image_processor = TilingImageProcessor(
                self.image_processor, tiling_cfg
            )

    preprocessor.EncoderPreprocessor.__init__ = init

    print(
        f"[fake_model] patched: dynamo.sglang={importlib.util.find_spec('dynamo.sglang').origin} "
        f"sglang={sglang.__file__} tile={TILE} tiling={tiling_cfg} weights={WEIGHTS}",
        flush=True,
    )


if os.environ.get("DYN_FAKE_ENCODER") == "1":
    apply()
