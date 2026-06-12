"""Sensor configuration for the event camera simulator."""

from dataclasses import dataclass


@dataclass
class SensorConfig:
    """Physical parameters of the simulated DVS pixel array.

    All rates are per pixel. Thresholds are natural-log intensity
    contrasts (a threshold of 0.3 fires after a ~35 % intensity change).
    """

    # Nominal contrast thresholds for ON (brightening) and OFF (darkening)
    # events. Real sensors are typically biased between 0.1 and 0.5.
    threshold_on: float = 0.3
    threshold_off: float = 0.3

    # Std-dev of the per-pixel Gaussian threshold mismatch (fixed-pattern
    # noise). Measured values for DVS pixels are around 0.02-0.04.
    sigma_threshold: float = 0.03

    # Photoreceptor 3 dB bandwidth in Hz at full brightness. The effective
    # bandwidth scales down with pixel intensity (dark pixels respond more
    # slowly). Set to 0 for an ideal, infinitely fast photoreceptor.
    cutoff_hz: float = 300.0

    # Minimum time between two events from the same pixel, in seconds.
    refractory_period_s: float = 0.5e-3

    # Rate of spontaneous ON "leak" events caused by the reset-switch leak
    # current, in Hz per pixel. Typical: 0.05-0.2 Hz at room temperature.
    leak_rate_hz: float = 0.1

    # Background-activity shot noise rate in Hz per pixel in darkness.
    # The effective rate decreases with increasing pixel brightness.
    shot_noise_rate_hz: float = 0.1

    # Fraction of pixels that are "hot" (defective, very noisy) and the
    # factor by which their shot-noise rate is multiplied.
    hot_pixel_fraction: float = 0.001
    hot_pixel_rate_multiplier: float = 100.0

    # Std-dev of Gaussian timestamp jitter in seconds (0 disables it).
    time_jitter_s: float = 0.0

    # Intensity (DN, 0-255 scale) below which the log response becomes
    # linear, modelling the photodiode's linear regime at low photocurrent.
    lin_log_threshold: float = 20.0

    def __post_init__(self):
        if self.threshold_on <= 0 or self.threshold_off <= 0:
            raise ValueError("contrast thresholds must be positive")
        if self.refractory_period_s < 0:
            raise ValueError("refractory period must be >= 0")

    @classmethod
    def dvs346(cls) -> "SensorConfig":
        """Parameters resembling a DAVIS346 under normal indoor biasing."""
        return cls()

    @classmethod
    def clean(cls) -> "SensorConfig":
        """An idealised noise-free sensor, useful for testing."""
        return cls(
            sigma_threshold=0.0,
            cutoff_hz=0.0,
            refractory_period_s=0.0,
            leak_rate_hz=0.0,
            shot_noise_rate_hz=0.0,
            hot_pixel_fraction=0.0,
            time_jitter_s=0.0,
        )
