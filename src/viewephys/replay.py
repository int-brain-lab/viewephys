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
    destination: str | Path | None = None,
    speed: float = 1.0,
    chunk_s: float = 0.25,
    stop: threading.Event | None = None,
) -> Path:
    """Writes source .bin to destination in chunks.

    Args:
        source (str | Path): Source .bin file to write
        destination (str | Path, optional): Destination directory to write to.
            Defaults to a `replay` subfolder next to the source.
        speed (float, optional): Speed of writing. Defaults to 1.0.
        chunk_s (float, optional): Chunk size in seconds. Defaults to 1.0.
        stop (threading.Event | None, optional): Event to signal stopping.
            Defaults to None.

    Returns:
        dest_bin: Path to written file

    Raises:
        ValueError: If the destination file is the source file, which would truncate it.
    """
    source = Path(source)
    meta = source.with_suffix(".meta")
    dest_bin = replay_path(source, destination)
    destination = dest_bin.parent
    if dest_bin.resolve() == source.resolve():
        raise ValueError(f"destination would overwrite the source file: {source}")
    destination.mkdir(parents=True, exist_ok=True)

    sr = spikeglx.Reader(source)
    shutil.copy(meta, destination / meta.name)

    chunk = int(sr.fs * chunk_s)
    t0 = time.monotonic()
    with open(dest_bin, "wb") as f:
        for i, start in enumerate(range(0, sr.ns, chunk), 1):
            if stop is not None and stop.is_set():
                break
            f.write(sr._raw[start : start + chunk].tobytes())
            f.flush()
            if (
                speed > 0
            ):  # sleep until the chunk's target time, so write time doesn't accumulate
                time.sleep(max(0.0, t0 + i * chunk_s / speed - time.monotonic()))

    return dest_bin


def replay_path(source: str | Path, destination: str | Path | None = None) -> Path:
    """Path that :func:`write_replay` will write for this source."""
    source = Path(source)
    return (
        Path(destination or source.parent / "replay") / source.with_suffix(".bin").name
    )
