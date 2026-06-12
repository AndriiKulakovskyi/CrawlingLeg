"""Event container with save/load helpers."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EventPacket:
    """A set of DVS events.

    Attributes:
        t: timestamps in microseconds, int64, sorted ascending.
        x: column coordinates, int32.
        y: row coordinates, int32.
        p: polarity, int8, +1 for ON (brightening), -1 for OFF.
    """

    t: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    x: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    y: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    p: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))

    def __len__(self) -> int:
        return self.t.shape[0]

    def __iter__(self):
        """Iterate events as (t_us, x, y, p) tuples."""
        return zip(self.t, self.x, self.y, self.p)

    @staticmethod
    def concatenate(packets) -> "EventPacket":
        packets = [pk for pk in packets if len(pk) > 0]
        if not packets:
            return EventPacket()
        out = EventPacket(
            t=np.concatenate([pk.t for pk in packets]),
            x=np.concatenate([pk.x for pk in packets]),
            y=np.concatenate([pk.y for pk in packets]),
            p=np.concatenate([pk.p for pk in packets]),
        )
        out.sort()
        return out

    def sort(self) -> None:
        """Sort events by timestamp (stable)."""
        order = np.argsort(self.t, kind="stable")
        self.t = self.t[order]
        self.x = self.x[order]
        self.y = self.y[order]
        self.p = self.p[order]

    def slice_time(self, t_start_us: int, t_end_us: int) -> "EventPacket":
        """Return events with t in [t_start_us, t_end_us)."""
        i0, i1 = np.searchsorted(self.t, [t_start_us, t_end_us])
        return EventPacket(self.t[i0:i1], self.x[i0:i1], self.y[i0:i1], self.p[i0:i1])

    def save_npz(self, path: str) -> None:
        np.savez_compressed(path, t=self.t, x=self.x, y=self.y, p=self.p)

    @staticmethod
    def load_npz(path: str) -> "EventPacket":
        data = np.load(path)
        return EventPacket(t=data["t"], x=data["x"], y=data["y"], p=data["p"])

    def save_txt(self, path: str) -> None:
        """Write one `t_us x y p` line per event (p as 1/0)."""
        cols = np.column_stack(
            [self.t, self.x, self.y, (self.p > 0).astype(np.int64)]
        )
        np.savetxt(path, cols, fmt="%d")
