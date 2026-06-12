"""Demo: simulate a DVS346-like event camera viewing synthetic scenes.

Runs the simulator on a rotating pinwheel disk and a moving bar, prints
event statistics, and writes visualisations + raw events to
examples/output/.

Usage:  python examples/demo_event_simulation.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from event_camera_simulator import (  # noqa: E402
    EventCameraSimulator,
    EventPacket,
    SensorConfig,
    events_to_image,
    save_png,
    scenes,
    time_surface,
)

WIDTH, HEIGHT = 240, 180
FPS = 2000.0
DURATION = 0.25
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_scene(name, frame_gen, out_dir):
    sim = EventCameraSimulator(WIDTH, HEIGHT, config=SensorConfig.dvs346(),
                               seed=42)
    start = time.time()
    packets = [sim.simulate_frame(frame, t) for frame, t in frame_gen]
    events = EventPacket.concatenate(packets)
    elapsed = time.time() - start

    n_on = int(np.sum(events.p > 0))
    n_off = len(events) - n_on
    rate = len(events) / DURATION
    print(f"\n=== {name} ===")
    print(f"  frames simulated : {len(packets)} @ {FPS:.0f} fps "
          f"({DURATION * 1e3:.0f} ms)")
    print(f"  events generated : {len(events):,} "
          f"(ON {n_on:,} / OFF {n_off:,})")
    print(f"  mean event rate  : {rate / 1e6:.2f} Mev/s "
          f"({rate / (WIDTH * HEIGHT):.1f} ev/px/s)")
    print(f"  wall time        : {elapsed:.2f} s")

    # Accumulated event images over consecutive 20 ms windows.
    window_us = 20_000
    for i in range(3):
        window = events.slice_time(i * window_us, (i + 1) * window_us)
        img = events_to_image(window, HEIGHT, WIDTH)
        path = os.path.join(out_dir, f"{name}_events_{i:02d}.png")
        save_png(img, path)
        print(f"  wrote {path}  ({len(window):,} events in window {i})")

    surf = (time_surface(events, HEIGHT, WIDTH) * 255).astype(np.uint8)
    surf_path = os.path.join(out_dir, f"{name}_time_surface.png")
    save_png(surf, surf_path)
    print(f"  wrote {surf_path}")

    npz_path = os.path.join(out_dir, f"{name}_events.npz")
    events.save_npz(npz_path)
    print(f"  wrote {npz_path}")
    return events


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_scene(
        "rotating_disk",
        scenes.rotating_disk(WIDTH, HEIGHT, FPS, DURATION, rev_per_s=4.0),
        OUT_DIR,
    )
    run_scene(
        "moving_bar",
        scenes.moving_bar(WIDTH, HEIGHT, FPS, DURATION, speed_px_s=600.0),
        OUT_DIR,
    )
    print(f"\nAll outputs are in {OUT_DIR}")


if __name__ == "__main__":
    main()
