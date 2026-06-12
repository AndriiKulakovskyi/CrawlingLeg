"""Synthetic high-frame-rate test scenes.

Each generator yields (frame, t_seconds) pairs with frames as float64
arrays on a 0-255 scale. Use a high fps (>= 1 kHz) so that inter-frame
motion stays small and the simulator's temporal interpolation is accurate.
"""

import numpy as np


def rotating_disk(width: int, height: int, fps: float, duration: float,
                  rev_per_s: float = 2.0, num_sectors: int = 8):
    """A pinwheel disk rotating about the image centre."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    angle = np.arctan2(yy - cy, xx - cx)
    radius = np.hypot(xx - cx, yy - cy)
    disk = radius < 0.45 * min(width, height)

    n_frames = int(round(fps * duration))
    for i in range(n_frames + 1):
        t = i / fps
        phase = 2.0 * np.pi * rev_per_s * t
        sectors = np.sin(num_sectors * (angle + phase)) > 0
        frame = np.where(disk & sectors, 220.0, np.where(disk, 35.0, 128.0))
        yield frame, t


def moving_bar(width: int, height: int, fps: float, duration: float,
               speed_px_s: float = 200.0, bar_width: int = 12,
               background: float = 30.0, foreground: float = 220.0):
    """A bright vertical bar sweeping left to right (wraps around)."""
    xx = np.arange(width, dtype=np.float64)
    n_frames = int(round(fps * duration))
    for i in range(n_frames + 1):
        t = i / fps
        x0 = (speed_px_s * t) % width
        in_bar = ((xx - x0) % width) < bar_width
        frame = np.where(in_bar, foreground, background)
        yield np.broadcast_to(frame, (height, width)).copy(), t


def drifting_grating(width: int, height: int, fps: float, duration: float,
                     spatial_period_px: float = 32.0,
                     speed_px_s: float = 100.0, contrast: float = 0.8):
    """A horizontally drifting sinusoidal grating."""
    xx = np.arange(width, dtype=np.float64)
    n_frames = int(round(fps * duration))
    for i in range(n_frames + 1):
        t = i / fps
        phase = 2.0 * np.pi * (xx - speed_px_s * t) / spatial_period_px
        row = 127.5 * (1.0 + contrast * np.sin(phase))
        yield np.broadcast_to(row, (height, width)).copy(), t
