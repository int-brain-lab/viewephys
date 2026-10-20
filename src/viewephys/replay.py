"""Replay SpikeGLX recordings by growing a copy.

Writes the source .bin to a destination file chunk by chunk, sleeping between writes.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import spikeglx

if TYPE_CHECKING:
    import threading


def write_replay(
    source: str | Path,
    destination: str | Path,
    speed: float = 1.0,
    chunk_s: float = 0.25,
    stop: threading.Event | None = None,
) -> Path:
    """Writes source .bin to destination in chunks.

    Args:
        source (str | Path): Source .bin file to write
        destination (str | Path): Destination directory to write to
        speed (float, optional): Speed of writing. Defaults to 1.0.
        chunk_s (float, optional): Chunk size in seconds. Defaults to 1.0.
        stop (threading.Event | None, optional): Event to signal stopping.
            Defaults to None.

    Returns:
        dest_bin: Path to written file
    """
    source = Path(source)
    meta = source.with_suffix(".meta")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    sr = spikeglx.Reader(source)
    dest_bin = replay_path(source, destination)
    shutil.copy(meta, destination / meta.name)

    chunk = int(sr.fs * chunk_s)
    with open(dest_bin, "wb") as f:
        for start in range(0, sr.ns, chunk):
            if stop is not None and stop.is_set():
                break
            f.write(sr._raw[start : start + chunk].tobytes())
            f.flush()
            if speed > 0:
                time.sleep(chunk_s / speed)

    return dest_bin


def replay_path(source: str | Path, destination: str | Path) -> Path:
    """Path that :func:`write_replay` will write for this source."""
    return Path(destination) / Path(source).with_suffix(".bin").name
