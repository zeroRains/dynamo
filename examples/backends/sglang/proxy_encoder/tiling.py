import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import msgspec
import numpy as np
import torch
from fake_encoder import TILE, TILE_GRID
from PIL import Image

PATCH_SIZE = 14  # Qwen patch size

# Images with area > TILE_MAX_AREA are cut into tiles plus the whole image; smaller ones stay whole.
TILE_MAX_AREA = 512 * 512

# max_log_ar = log(max(1, _MIN_AR_CAP, source aspect ratio) * _AR_TOLERANCE)
_MIN_AR_CAP = 2.0
_AR_TOLERANCE = 1.05


class TilePlan(msgspec.Struct, frozen=True):
    src_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    grid: Tuple[int, int]  # (cols, rows)
    tile_size: Tuple[int, int]
    boxes: List[Tuple[int, int, int, int]]
    native: bool  # True: area <= max_area, the whole image is the single tile


def _aspect_penalty(w: float, h: float) -> float:
    return abs(math.log(w / h))


def _align_tile_size(w: float, h: float, patch: int, max_area: int) -> Tuple[int, int]:
    aw = max(patch, int(round(w / patch)) * patch)
    ah = max(patch, int(round(h / patch)) * patch)
    while aw * ah > max_area:
        if aw >= ah and aw > patch:
            aw -= patch
        elif ah > patch:
            ah -= patch
        else:
            break
    return aw, ah


def _search_grid(
    width: int,
    height: int,
    max_area: int,
    patch: int,
    max_log_ar: float,
    max_tiles: int,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    lower_bound = max(1, math.ceil(width * height / max_area))
    for n in range(lower_bound, max_tiles + 1):
        best, best_key = None, None
        for rows in range(1, n + 1):
            if n % rows:
                continue
            cols = n // rows
            if cols * patch > width or rows * patch > height:
                continue
            tw, th = _align_tile_size(width / cols, height / rows, patch, max_area)
            if tw * th > max_area:
                continue
            pen = _aspect_penalty(tw, th)
            if pen > max_log_ar:
                continue
            distort = abs(math.log(cols * tw / width)) + abs(
                math.log(rows * th / height)
            )
            key = (round(pen, 6), round(distort, 6), abs(rows - cols))
            if best_key is None or key < best_key:
                best_key, best = key, ((cols, rows), (tw, th))
        if best is not None:
            return best
    return None


def plan_tiles(
    width: int, height: int, max_area: int, patch_size: int, max_tiles: int
) -> TilePlan:
    """Equal, patch-aligned tiles of area <= max_area, cut row-major from the image resized to cols*tw x rows*th.

    Tile count: smallest feasible value >= ceil(W*H/max_area), capped at max_tiles (the image is shrunk first if exceeded).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid size: {width}x{height}")

    if width * height <= max_area:
        tw, th = _align_tile_size(width, height, patch_size, max_area)
        return TilePlan(
            src_size=(width, height),
            resized_size=(tw, th),
            grid=(1, 1),
            tile_size=(tw, th),
            boxes=[(0, 0, tw, th)],
            native=True,
        )

    src_ar = max(width / height, height / width)
    max_log_ar = math.log(max(1.0, _MIN_AR_CAP, src_ar) * _AR_TOLERANCE)

    def _try(w: int, h: int):
        # Relax the aspect cap step by step instead of failing when alignment rules out every candidate.
        for relax in (1.0, 1.5, 2.0, 3.0, 1e9):
            got = _search_grid(
                w, h, max_area, patch_size, max_log_ar * relax, max_tiles
            )
            if got is not None:
                return got
        return None

    found = _try(width, height)
    if found is None:
        scale = math.sqrt(max_tiles * max_area / float(width * height))
        for _ in range(40):
            w_eff = max(patch_size, int(width * scale))
            h_eff = max(patch_size, int(height * scale))
            found = _try(w_eff, h_eff)
            if found is not None:
                break
            scale *= 0.95
        if found is None:
            found = (
                (1, 1),
                _align_tile_size(width * scale, height * scale, patch_size, max_area),
            )

    (cols, rows), (tw, th) = found
    boxes = [
        (c * tw, r * th, (c + 1) * tw, (r + 1) * th)
        for r in range(rows)
        for c in range(cols)
    ]
    return TilePlan(
        src_size=(width, height),
        resized_size=(cols * tw, rows * th),
        grid=(cols, rows),
        tile_size=(tw, th),
        boxes=boxes,
        native=False,
    )


def should_tile(image: Image.Image) -> bool:
    """Tile only images larger than TILE_MAX_AREA."""
    return image.width * image.height > TILE_MAX_AREA


def split_image(
    img: Image.Image, max_area: int, patch_size: int, max_tiles: int
) -> List[Image.Image]:
    plan = plan_tiles(img.width, img.height, max_area, patch_size, max_tiles)
    if (img.width, img.height) != plan.resized_size:
        img = img.resize(plan.resized_size, Image.BICUBIC)
    return [img.crop(b) for b in plan.boxes]


def _env_bool(name, default):
    return os.environ.get(name, default).strip().lower() in ("true", "1")


@dataclass(frozen=True)
class TilingConfig:
    enable: bool = True
    enable_multi_image: bool = True
    max_tiles: int = 16
    max_tiles_multi_image: int = 9

    def __post_init__(self):
        for name, cap in (
            ("FAKE_MAX_TILES", self.max_tiles),
            ("FAKE_MAX_TILES_MULTI_IMAGE", self.max_tiles_multi_image),
        ):
            if cap < 1:
                raise ValueError(f"{name} must be >= 1, got {cap}")

    @classmethod
    def from_env(cls):
        return cls(
            enable=_env_bool("FAKE_ENABLE_TILING", "true"),
            enable_multi_image=_env_bool("FAKE_ENABLE_TILING_MULTI_IMAGE", "true"),
            max_tiles=int(os.environ.get("FAKE_MAX_TILES", "16")),
            max_tiles_multi_image=int(
                os.environ.get("FAKE_MAX_TILES_MULTI_IMAGE", "9")
            ),
        )


def to_rgb_white_background(image):
    """Convert to RGB, compositing transparency onto white; I;16 is scaled down to L first."""
    try:
        if image.mode == "I;16":
            # I-mode point() only takes add/sub/mul, so this must stay a multiply.
            image = image.point(lambda i: i * (1 / 256)).convert("L")
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            if rgba.getchannel("A").getextrema()[0] < 255:
                background = Image.new("RGBA", rgba.size, (255, 255, 255))
                background.paste(rgba, (0, 0), rgba)
                image = background
    except Exception:
        # Fall back to the plain conversion.
        pass
    return image.convert("RGB")


def tile_items(image, cfg, *, n_images=1):
    """One image -> list of 448x448 items.

    Tiled: row-major tiles, then the whole image (global view). Not tiled: the whole image only.
    n_images picks the single/multi-image max_tiles and whether tiling is allowed.
    """
    image = to_rgb_white_background(image)
    single = n_images <= 1
    tiling_on = cfg.enable and (single or cfg.enable_multi_image)
    max_tiles = cfg.max_tiles if single else cfg.max_tiles_multi_image
    if tiling_on and should_tile(image):
        items = split_image(image, TILE_MAX_AREA, PATCH_SIZE, max_tiles) + [image]
    else:
        items = [image]
    return [i.resize((TILE, TILE), Image.Resampling.BICUBIC) for i in items]


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if x and isinstance(x[0], (torch.Tensor, np.ndarray)):
        return torch.cat([_as_tensor(i) for i in x], dim=0)
    return torch.tensor(x)


def _to_pil(img):
    """PIL as is; ndarray as HWC; Tensor as CHW in 0-255."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    raise TypeError(f"unsupported image type: {type(img)}")


class TilingImageProcessor:
    """Wrap the HF image processor: each image becomes n items whose grids merge into [1, 32, 32n]."""

    def __init__(self, base, cfg):
        self._base = base
        self._cfg = cfg

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def __call__(self, images=None, **kw):
        if not isinstance(images, (list, tuple)):
            images = [images]
        items, counts = [], []
        for img in images:
            its = tile_items(_to_pil(img), self._cfg, n_images=len(images))
            items.extend(its)
            counts.append(len(its))

        out = self._base(images=items, **kw)
        pixel = _as_tensor(out["pixel_values"])
        grid = _as_tensor(out["image_grid_thw"]).to(pixel.device)
        expect = torch.tensor(
            [1, TILE_GRID, TILE_GRID], dtype=grid.dtype, device=grid.device
        )
        assert grid.shape == (len(items), 3) and bool((grid == expect).all()), (
            grid.tolist()
        )

        merged = torch.tensor(
            [[1, TILE_GRID, TILE_GRID * n] for n in counts],
            dtype=torch.long,
            device=pixel.device,
        )
        assert pixel.shape[0] == int(merged.prod(1).sum()), (
            pixel.shape,
            merged.tolist(),
        )
        out["pixel_values"], out["image_grid_thw"] = pixel, merged
        return out
