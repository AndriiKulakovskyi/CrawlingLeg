import numpy as np
import pytest

from event_camera_simulator import (
    EventCameraSimulator,
    EventPacket,
    SensorConfig,
    events_to_image,
    events_to_voxel_grid,
    lin_log,
    scenes,
    time_surface,
)


def make_clean_sim(w=8, h=8, **overrides):
    cfg = SensorConfig.clean()
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return EventCameraSimulator(w, h, config=cfg, seed=0)


def test_static_scene_produces_no_events():
    sim = make_clean_sim()
    frame = np.full((8, 8), 100.0)
    packets = [sim.simulate_frame(frame, i * 1e-3) for i in range(20)]
    assert sum(len(p) for p in packets) == 0


def test_brightening_pixel_emits_on_events():
    sim = make_clean_sim()
    f0 = np.full((8, 8), 50.0)
    sim.simulate_frame(f0, 0.0)
    f1 = f0.copy()
    f1[3, 4] = 150.0  # log contrast ~ ln(151/51) = 1.085 -> 3 events at 0.3
    ev = sim.simulate_frame(f1, 1e-3)
    assert len(ev) == 3
    assert np.all(ev.p == 1)
    assert np.all(ev.x == 4) and np.all(ev.y == 3)
    assert np.all((ev.t > 0) & (ev.t <= 1000))


def test_darkening_pixel_emits_off_events():
    sim = make_clean_sim()
    f0 = np.full((8, 8), 150.0)
    sim.simulate_frame(f0, 0.0)
    f1 = f0.copy()
    f1[0, 0] = 50.0
    ev = sim.simulate_frame(f1, 1e-3)
    assert len(ev) == 3
    assert np.all(ev.p == -1)


def test_subthreshold_change_is_memorised_not_lost():
    sim = make_clean_sim()
    f0 = np.full((8, 8), 100.0)
    sim.simulate_frame(f0, 0.0)
    # Two half-threshold steps must add up to one event.
    f1 = np.full((8, 8), 100.0 * np.exp(0.2))
    assert len(sim.simulate_frame(f1, 1e-3)) == 0
    f2 = np.full((8, 8), 100.0 * np.exp(0.4))
    ev = sim.simulate_frame(f2, 2e-3)
    assert len(ev) == 64  # every pixel fires exactly once
    assert np.all(ev.p == 1)


def test_refractory_period_limits_event_rate():
    sim = make_clean_sim(refractory_period_s=1e-3)
    f0 = np.full((8, 8), 20.0)
    sim.simulate_frame(f0, 0.0)
    f1 = np.full((8, 8), 250.0)  # huge jump, many threshold crossings
    ev = sim.simulate_frame(f1, 1e-3)
    counts = np.bincount(ev.y * 8 + ev.x, minlength=64)
    assert counts.max() <= 1


def test_event_timestamps_interpolated_within_interval():
    sim = make_clean_sim(4, 4)
    f0 = np.full((4, 4), 50.0)
    sim.simulate_frame(f0, 0.0)
    f1 = np.full((4, 4), 150.0)
    ev = sim.simulate_frame(f1, 0.01)
    assert len(ev) > 0
    assert np.all(np.diff(ev.t) >= 0)
    assert np.all((ev.t > 0) & (ev.t <= 10_000))
    # Multiple events per pixel must have distinct interpolated times.
    px = ev.t[(ev.x == 0) & (ev.y == 0)]
    assert len(np.unique(px)) == len(px)


def test_leak_generates_on_events_on_static_scene():
    sim = make_clean_sim(leak_rate_hz=5.0)
    frame = np.full((8, 8), 100.0)
    duration, fps = 2.0, 100
    packets = [sim.simulate_frame(frame, i / fps)
               for i in range(int(duration * fps) + 1)]
    ev = EventPacket.concatenate(packets)
    assert len(ev) > 0
    assert np.all(ev.p == 1)
    # Expected ~ 5 Hz * 2 s * 64 px = 640, with +-50% per-pixel rate spread.
    assert 300 < len(ev) < 1100


def test_shot_noise_rate_and_polarity():
    sim = make_clean_sim(16, 16, shot_noise_rate_hz=10.0)
    frame = np.zeros((16, 16))  # darkness: full shot-noise rate
    duration, fps = 2.0, 200
    packets = [sim.simulate_frame(frame, i / fps)
               for i in range(int(duration * fps) + 1)]
    ev = EventPacket.concatenate(packets)
    expected = 10.0 * duration * 16 * 16
    assert 0.7 * expected < len(ev) < 1.3 * expected
    on_fraction = np.mean(ev.p > 0)
    assert 0.4 < on_fraction < 0.6


def test_shot_noise_is_reduced_in_bright_scenes():
    counts = []
    for level in (0.0, 255.0):
        sim = make_clean_sim(16, 16, shot_noise_rate_hz=10.0)
        frame = np.full((16, 16), level)
        packets = [sim.simulate_frame(frame, i / 200) for i in range(401)]
        counts.append(len(EventPacket.concatenate(packets)))
    assert counts[1] < 0.5 * counts[0]


def test_threshold_mismatch_is_per_pixel():
    sim = EventCameraSimulator(
        32, 32, config=SensorConfig(sigma_threshold=0.03), seed=1)
    assert np.std(sim.theta_on) > 0.01
    assert np.all(sim.theta_on > 0)


def test_hot_pixels_have_elevated_rate():
    cfg = SensorConfig(shot_noise_rate_hz=1.0, hot_pixel_fraction=0.01,
                       hot_pixel_rate_multiplier=50.0)
    sim = EventCameraSimulator(100, 100, config=cfg, seed=2)
    n_hot = np.sum(sim.shot_rate > 1.0)
    assert n_hot == 100
    assert sim.shot_rate.max() == pytest.approx(50.0)


def test_lin_log_continuous_and_monotonic():
    x = np.linspace(0, 255, 4096)
    y = lin_log(x)
    assert np.all(np.diff(y) > 0)
    eps = 1e-6
    assert lin_log(np.array([20.0 - eps]))[0] == pytest.approx(
        lin_log(np.array([20.0 + eps]))[0], abs=1e-4)


def test_simulate_video_matches_frame_by_frame():
    frames = list(scenes.moving_bar(32, 24, fps=1000, duration=0.05))
    sim = EventCameraSimulator(32, 24, config=SensorConfig.clean(), seed=0)
    ev = sim.simulate_video((f for f, _ in frames), (t for _, t in frames))
    assert len(ev) > 0
    assert np.all(np.diff(ev.t) >= 0)
    assert ev.x.min() >= 0 and ev.x.max() < 32
    assert ev.y.min() >= 0 and ev.y.max() < 24
    assert set(np.unique(ev.p)) <= {-1, 1}


def test_realistic_config_runs_on_rotating_disk():
    sim = EventCameraSimulator(48, 48, config=SensorConfig.dvs346(), seed=3)
    frames = scenes.rotating_disk(48, 48, fps=1000, duration=0.05)
    ev = EventPacket.concatenate(
        [sim.simulate_frame(f, t) for f, t in frames])
    assert len(ev) > 100
    assert np.any(ev.p == 1) and np.any(ev.p == -1)


def test_non_increasing_timestamp_rejected():
    sim = make_clean_sim()
    frame = np.zeros((8, 8))
    sim.simulate_frame(frame, 0.0)
    with pytest.raises(ValueError):
        sim.simulate_frame(frame, 0.0)


def test_wrong_frame_shape_rejected():
    sim = make_clean_sim()
    with pytest.raises(ValueError):
        sim.simulate_frame(np.zeros((4, 4)), 0.0)


def test_event_packet_slice_save_load(tmp_path):
    sim = make_clean_sim()
    frames = scenes.moving_bar(8, 8, fps=1000, duration=0.02)
    ev = EventPacket.concatenate(
        [sim.simulate_frame(f, t) for f, t in frames])
    sl = ev.slice_time(0, 10_000)
    assert np.all(sl.t < 10_000)
    path = tmp_path / "events.npz"
    ev.save_npz(str(path))
    loaded = EventPacket.load_npz(str(path))
    assert np.array_equal(loaded.t, ev.t)
    assert np.array_equal(loaded.p, ev.p)


def test_visualization_shapes():
    sim = make_clean_sim(16, 12)
    frames = scenes.moving_bar(16, 12, fps=1000, duration=0.02)
    ev = EventPacket.concatenate(
        [sim.simulate_frame(f, t) for f, t in frames])
    img = events_to_image(ev, 12, 16)
    assert img.shape == (12, 16, 3) and img.dtype == np.uint8
    assert img.sum() > 0
    surf = time_surface(ev, 12, 16)
    assert surf.shape == (12, 16)
    assert surf.max() <= 1.0
    grid = events_to_voxel_grid(ev, 5, 12, 16)
    assert grid.shape == (5, 12, 16)
    # Total polarity mass is preserved by the bilinear binning.
    assert grid.sum() == pytest.approx(ev.p.astype(np.float64).sum())
