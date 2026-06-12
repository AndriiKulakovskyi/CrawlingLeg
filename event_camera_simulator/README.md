# Event Camera Simulator

A realistic simulator for event-based cameras (dynamic vision sensors such
as the DVS128, DAVIS240, DAVIS346 or Prophesee sensors). It converts a
sequence of intensity frames into an asynchronous stream of brightness
events `(t, x, y, polarity)`, modelling the dominant physical effects of
real DVS pixels. Only NumPy is required.

## Sensor model

Each pixel follows the standard DVS pixel pipeline (in the spirit of the
v2e and ESIM simulators):

1. **Log photoreceptor** — intensity is mapped through `lin_log`: linear
   below a junction intensity (the photodiode's linear regime in the dark)
   and logarithmic above, with a continuous joint.
2. **Finite photoreceptor bandwidth** — the log signal passes through a
   first-order low-pass filter whose cutoff scales with pixel brightness,
   so dark pixels respond sluggishly and smear fast motion, exactly as in
   real sensors.
3. **Threshold comparison with mismatch** — an event fires whenever the
   filtered log intensity differs from a memorized baseline by more than
   the contrast threshold. ON and OFF thresholds are drawn per pixel from
   a Gaussian (fixed-pattern noise), frozen at construction.
4. **Sub-frame timestamps** — when a frame interval contains several
   threshold crossings, each event gets a linearly interpolated timestamp,
   so the output stream has genuine sub-frame timing resolution.
5. **Refractory period** — a pixel that just fired ignores further
   crossings for a configurable dead time; crossings during the dead time
   still reset the baseline but the events are lost.
6. **Leak events** — the reset-switch leak current makes the baseline
   drift downward, producing the spontaneous ON background events seen on
   every real DVS, with per-pixel rate variation.
7. **Shot noise** — random ON/OFF background activity whose rate grows in
   the dark, including a small population of *hot pixels* with strongly
   elevated rates.
8. **Optional timestamp jitter** — Gaussian noise on event timestamps.

## Usage

```python
import numpy as np
from event_camera_simulator import EventCameraSimulator, SensorConfig, scenes

sim = EventCameraSimulator(240, 180, config=SensorConfig.dvs346(), seed=42)

# Feed frames (grayscale, 0-255 scale) at their timestamps in seconds.
# Use a high frame rate (>= 1 kHz) so inter-frame motion stays small.
events = sim.simulate_video(
    *zip(*scenes.rotating_disk(240, 180, fps=2000, duration=0.25)))

print(len(events), "events")          # (t [us], x, y, p in {-1, +1})
events.save_npz("events.npz")         # or events.save_txt("events.txt")
```

Frame-by-frame streaming is also supported:

```python
for frame, t in my_video_source():
    packet = sim.simulate_frame(frame, t)
    process(packet)
```

`SensorConfig` exposes all physical parameters (thresholds, mismatch,
bandwidth, refractory period, leak/shot-noise rates, hot pixels,
timestamp jitter). `SensorConfig.clean()` gives an idealised noise-free
sensor for algorithm debugging.

## Visualisation and representations

```python
from event_camera_simulator import events_to_image, time_surface, \
    events_to_voxel_grid, save_png

img = events_to_image(events.slice_time(0, 20_000), 180, 240)  # ON red / OFF blue
save_png(img, "events.png")
surf = time_surface(events, 180, 240, tau_us=30_000)           # exponential decay
grid = events_to_voxel_grid(events, num_bins=5, height=180, width=240)
```

## Demo and tests

```bash
python examples/demo_event_simulation.py   # writes PNGs + .npz to examples/output/
python -m pytest tests/                    # 18 tests covering the sensor model
```

Example output (rotating pinwheel, 20 ms accumulation): the sector edges
generate interleaved ON/OFF wavefronts while leak and shot noise sprinkle
isolated events over the static background.

## References

* Lichtsteiner, Posch, Delbruck, *A 128x128 120 dB 15 µs Latency
  Asynchronous Temporal Contrast Vision Sensor*, JSSC 2008.
* Hu, Liu, Delbruck, *v2e: From Video Frames to Realistic DVS Events*,
  CVPRW 2021.
* Rebecq, Gehrig, Scaramuzza, *ESIM: an Open Event Camera Simulator*,
  CoRL 2018.
