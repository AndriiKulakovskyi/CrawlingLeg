"""Rendering helpers for event streams (no third-party image deps)."""

import struct
import zlib

import numpy as np

from .events import EventPacket


def events_to_image(events: EventPacket, height: int, width: int) -> np.ndarray:
    """Render events as an RGB image: ON = red, OFF = blue, black background.

    Brightness scales with the per-pixel event count.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events) == 0:
        return img
    on = events.p > 0
    counts_on = np.zeros((height, width), np.int64)
    counts_off = np.zeros((height, width), np.int64)
    np.add.at(counts_on, (events.y[on], events.x[on]), 1)
    np.add.at(counts_off, (events.y[~on], events.x[~on]), 1)
    peak = max(counts_on.max(), counts_off.max(), 1)
    img[..., 0] = np.clip(counts_on * 255 // min(peak, 3), 0, 255)
    img[..., 2] = np.clip(counts_off * 255 // min(peak, 3), 0, 255)
    return img


def time_surface(events: EventPacket, height: int, width: int,
                 t_ref_us: int = None, tau_us: float = 30_000.0) -> np.ndarray:
    """Exponentially decayed map of the most recent event per pixel.

    Returns a (height, width) float array in [0, 1]; recent events are
    bright regardless of polarity.
    """
    surface = np.zeros((height, width), dtype=np.float64)
    if len(events) == 0:
        return surface
    if t_ref_us is None:
        t_ref_us = int(events.t[-1])
    last_t = np.full((height, width), -np.inf)
    # Events are time-sorted, so later writes win.
    last_t[events.y, events.x] = events.t
    valid = np.isfinite(last_t)
    surface[valid] = np.exp(-(t_ref_us - last_t[valid]) / tau_us)
    return surface


def events_to_voxel_grid(events: EventPacket, num_bins: int,
                         height: int, width: int) -> np.ndarray:
    """Accumulate polarity into a (num_bins, height, width) voxel grid.

    Each event's polarity is distributed between the two nearest temporal
    bins by linear interpolation (the standard representation used by
    event-based deep learning pipelines).
    """
    grid = np.zeros((num_bins, height, width), dtype=np.float64)
    if len(events) == 0:
        return grid
    t = events.t.astype(np.float64)
    t0, t1 = t[0], t[-1]
    tn = (t - t0) / max(t1 - t0, 1.0) * (num_bins - 1)
    lo = np.floor(tn).astype(np.int64)
    hi = np.minimum(lo + 1, num_bins - 1)
    w_hi = tn - lo
    pol = events.p.astype(np.float64)
    np.add.at(grid, (lo, events.y, events.x), pol * (1.0 - w_hi))
    np.add.at(grid, (hi, events.y, events.x), pol * w_hi)
    return grid


def save_png(image: np.ndarray, path: str) -> None:
    """Write a uint8 grayscale (H, W) or RGB (H, W, 3) array as a PNG."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    if image.ndim == 2:
        color_type, channels = 0, 1
    elif image.ndim == 3 and image.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError("image must be (H, W) or (H, W, 3)")

    height, width = image.shape[:2]
    raw = image.reshape(height, width * channels)
    # Prepend filter byte 0 (None) to each scanline.
    scanlines = np.concatenate(
        [np.zeros((height, 1), np.uint8), raw], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(scanlines, 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
