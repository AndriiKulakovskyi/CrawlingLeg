"""Realistic event-based (DVS) camera simulator.

Converts intensity frames into asynchronous brightness-change events,
modelling the dominant physical effects of real dynamic vision sensors
(DVS128 / DAVIS240 / DAVIS346 class devices):

* logarithmic photoreceptor response (linear below a junction point),
* finite, intensity-dependent photoreceptor bandwidth,
* per-pixel contrast-threshold mismatch (fixed-pattern noise),
* pixel refractory period,
* leak events (spontaneous ON events from the reset-switch leak current),
* shot-noise background activity that grows in the dark,
* hot pixels with strongly elevated noise rates.
"""

from .config import SensorConfig
from .events import EventPacket
from .simulator import EventCameraSimulator, lin_log
from .visualization import (
    events_to_image,
    events_to_voxel_grid,
    save_png,
    time_surface,
)
from . import scenes

__all__ = [
    "SensorConfig",
    "EventPacket",
    "EventCameraSimulator",
    "lin_log",
    "events_to_image",
    "events_to_voxel_grid",
    "time_surface",
    "save_png",
    "scenes",
]
