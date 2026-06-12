"""Core DVS pixel-array simulation.

The model follows the structure of established simulators (v2e, ESIM):
each pixel low-pass filters the log intensity, compares it against a
memorized baseline, and emits an event whenever the difference crosses a
per-pixel contrast threshold. Sub-frame event timestamps are recovered by
linearly interpolating the threshold-crossing times between frames.
"""

import numpy as np

from .config import SensorConfig
from .events import EventPacket


def lin_log(intensity: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Map intensity (0-255 scale) to log space, linear below `threshold`.

    Real photoreceptors respond linearly at low photocurrents and
    logarithmically above; the two branches are joined so the mapping is
    continuous at the junction.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    # Shift by 1 so zero intensity stays finite.
    shifted = intensity + 1.0
    junction = threshold + 1.0
    slope = np.log(junction) / junction
    return np.where(shifted <= junction, shifted * slope, np.log(shifted))


class EventCameraSimulator:
    """Stateful frame-to-events converter for one sensor.

    Feed frames in increasing time order with `simulate_frame`, or convert
    a whole sequence with `simulate_video`. For faithful results the input
    frame rate should be high enough that inter-frame motion is small
    (ideally >= 1 kHz; linear interpolation fills in sub-frame timing).
    """

    def __init__(self, width: int, height: int,
                 config: SensorConfig = None, seed: int = None):
        self.width = int(width)
        self.height = int(height)
        self.config = config if config is not None else SensorConfig()
        self.rng = np.random.default_rng(seed)

        cfg = self.config
        shape = (self.height, self.width)

        # Frozen per-pixel threshold mismatch (fixed-pattern noise).
        self.theta_on = np.maximum(
            self.rng.normal(cfg.threshold_on, cfg.sigma_threshold, shape), 0.01)
        self.theta_off = np.maximum(
            self.rng.normal(cfg.threshold_off, cfg.sigma_threshold, shape), 0.01)

        # Per-pixel leak rate: leak current varies strongly across the array.
        self.leak_rate = cfg.leak_rate_hz * self.rng.uniform(0.5, 1.5, shape)

        # Per-pixel shot-noise rate, with a hot-pixel population.
        self.shot_rate = np.full(shape, cfg.shot_noise_rate_hz)
        if cfg.hot_pixel_fraction > 0 and cfg.shot_noise_rate_hz > 0:
            n_hot = int(round(cfg.hot_pixel_fraction * self.height * self.width))
            if n_hot > 0:
                hot = self.rng.choice(self.height * self.width, n_hot,
                                      replace=False)
                self.shot_rate.ravel()[hot] *= cfg.hot_pixel_rate_multiplier

        # Pixel state, initialised on the first frame.
        self.lp_log = None        # photoreceptor output (low-passed log intensity)
        self.base_log = None      # memorized log intensity at last reset
        self.last_event_t = None  # per-pixel time of last event (s)
        self.t_prev = None

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Forget pixel state; the next frame re-initialises the sensor."""
        self.lp_log = None
        self.base_log = None
        self.last_event_t = None
        self.t_prev = None

    def simulate_frame(self, frame: np.ndarray, t: float) -> EventPacket:
        """Process one grayscale frame at time `t` (seconds).

        `frame` is (height, width), values on a 0-255 scale (any dtype).
        Returns the events generated since the previous frame, sorted by
        timestamp. The first frame only initialises state.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != (self.height, self.width):
            raise ValueError(
                f"frame shape {frame.shape} != ({self.height}, {self.width})")

        cfg = self.config
        log_frame = lin_log(frame, cfg.lin_log_threshold)

        if self.t_prev is None:
            self.lp_log = log_frame.copy()
            self.base_log = log_frame.copy()
            self.last_event_t = np.full(frame.shape, -np.inf)
            self.t_prev = float(t)
            return EventPacket()

        dt = float(t) - self.t_prev
        if dt <= 0:
            raise ValueError("frame timestamps must be strictly increasing")

        # --- Photoreceptor: intensity-dependent first-order low-pass. ----
        if cfg.cutoff_hz > 0:
            inten = np.clip(frame / 255.0, 0.0, 1.0)
            f3db = cfg.cutoff_hz * (0.05 + 0.95 * inten)
            alpha = 1.0 - np.exp(-dt * 2.0 * np.pi * f3db)
            self.lp_log += alpha * (log_frame - self.lp_log)
        else:
            self.lp_log = log_frame

        # --- Leak: baseline decays, producing spontaneous ON events. -----
        if cfg.leak_rate_hz > 0:
            self.base_log -= self.leak_rate * self.theta_on * dt

        signal = self._generate_signal_events(dt)
        noise = self._generate_shot_noise(frame, dt)
        self.t_prev = float(t)

        packet = EventPacket.concatenate([signal, noise])
        if cfg.time_jitter_s > 0 and len(packet) > 0:
            jitter = self.rng.normal(0.0, cfg.time_jitter_s * 1e6, len(packet))
            packet.t = np.clip(
                packet.t + jitter.astype(np.int64),
                int(self.t_prev * 1e6) - int(dt * 1e6), int(self.t_prev * 1e6))
            packet.sort()
        return packet

    def simulate_video(self, frames, timestamps) -> EventPacket:
        """Convert a frame sequence (iterables of frames and seconds)."""
        packets = [self.simulate_frame(f, t) for f, t in zip(frames, timestamps)]
        return EventPacket.concatenate(packets)

    # ------------------------------------------------------------------

    def _generate_signal_events(self, dt: float) -> EventPacket:
        """Threshold crossings of (photoreceptor - baseline) since t_prev."""
        cfg = self.config
        delta = self.lp_log - self.base_log
        pol = np.sign(delta)
        theta = np.where(pol > 0, self.theta_on, self.theta_off)

        n_events = np.floor(np.abs(delta) / theta).astype(np.int64)

        # The refractory period bounds how many events one pixel can emit
        # in this interval (and bounds the interpolation loop below).
        if cfg.refractory_period_s > 0:
            n_cap = max(1, int(dt / cfg.refractory_period_s) + 1)
            n_events = np.minimum(n_events, n_cap)

        max_n = int(n_events.max()) if n_events.size else 0
        if max_n == 0:
            return EventPacket()

        ts_list, x_list, y_list, p_list = [], [], [], []
        flat_delta = delta.ravel()
        flat_theta = theta.ravel()
        flat_pol = pol.ravel()
        flat_n = n_events.ravel()
        flat_last = self.last_event_t.ravel()
        flat_base = self.base_log.ravel()

        for k in range(1, max_n + 1):
            idx = np.flatnonzero(flat_n >= k)
            # Linear interpolation of the k-th threshold-crossing time.
            ts = self.t_prev + dt * (k * flat_theta[idx]) / np.abs(flat_delta[idx])

            # The baseline resets at every comparator crossing, even when
            # the pixel is refractory and the event itself is lost.
            flat_base[idx] += flat_pol[idx] * flat_theta[idx]

            keep = ts >= flat_last[idx] + cfg.refractory_period_s
            idx, ts = idx[keep], ts[keep]
            flat_last[idx] = ts

            ts_list.append(ts)
            y_list.append(idx // self.width)
            x_list.append(idx % self.width)
            p_list.append(flat_pol[idx])

        if not ts_list:
            return EventPacket()
        return EventPacket(
            t=np.round(np.concatenate(ts_list) * 1e6).astype(np.int64),
            x=np.concatenate(x_list).astype(np.int32),
            y=np.concatenate(y_list).astype(np.int32),
            p=np.concatenate(p_list).astype(np.int8),
        )

    def _generate_shot_noise(self, frame: np.ndarray, dt: float) -> EventPacket:
        """Background-activity events, more frequent in dark pixels."""
        cfg = self.config
        if cfg.shot_noise_rate_hz <= 0:
            return EventPacket()

        inten = np.clip(frame / 255.0, 0.0, 1.0)
        rate = self.shot_rate * (1.0 - 0.75 * inten)
        prob = np.clip(rate * dt, 0.0, 0.8)
        fired = np.flatnonzero(self.rng.random(frame.size) < prob.ravel())
        if fired.size == 0:
            return EventPacket()

        ts = self.t_prev + self.rng.uniform(0.0, dt, fired.size)
        pol = self.rng.choice(np.array([-1, 1], np.int8), fired.size)

        # A noise event resets the pixel to its current illumination, and
        # the refractory timer restarts.
        self.base_log.ravel()[fired] = self.lp_log.ravel()[fired]
        self.last_event_t.ravel()[fired] = ts

        return EventPacket(
            t=np.round(ts * 1e6).astype(np.int64),
            x=(fired % self.width).astype(np.int32),
            y=(fired // self.width).astype(np.int32),
            p=pol,
        )
