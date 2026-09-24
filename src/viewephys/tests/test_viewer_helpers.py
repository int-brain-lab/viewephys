from pathlib import Path

import numpy as np


def build_lfpack_h5(
    path: Path,
    recording: str,
    nc: int = 64,
    ns: int = 6000,
    fs: float = 250.0,
    annotate: bool = False,
    t0_sync: float | None = None,
) -> Path:
    """Write a tiny single-recording lfpack HDF5 file for tests.

    Requires the optional ``lfpack`` dependency; callers should guard with
    ``pytest.importorskip('lfpack')``.  When ``annotate`` is True, per-channel
    brain-region annotations (``atlas_id``/``acronym``) are stored so the
    brain-region code path can be exercised.  ``t0_sync`` stores a sync-clock
    origin (seconds) on the file, exercising the sync-data path of
    ``LFPackReader.t0``; omitted (``None``) reproduces a file with no sync
    data, where ``LFPackReader.t0`` is ``NaN``.
    """
    import lfpack

    rng = np.random.default_rng(0)
    data = (rng.standard_normal((ns, nc)) * 1e-5).astype(np.float32)
    npy = Path(path).with_suffix(".cadzow.npy")
    np.save(npy, data)
    channels = None
    if annotate:
        channels = {
            "atlas_id": np.zeros(nc, dtype=np.int32),  # 0 = root, valid in br.id
            "acronym": ["void"] * nc,
        }
    lfpack.compress_to_h5(
        npy,
        path,
        recording=recording,
        fs=fs,
        n_jobs=1,
        channels=channels,
        t0_sync=t0_sync,
    )
    return Path(path)


def ricker(points: int, a: float) -> np.ndarray:
    amplitude = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return amplitude * mod * gauss


def synthetic_seismic_data(
    ntr: int = 500, ns: int = 2000
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = np.zeros((ntr, ns), np.float32)
    data[:, 500:600] = ricker(100, 4)

    noise = np.random.randn(ntr, ns).astype(np.float32) / 10
    a, b = np.meshgrid(np.arange(ntr / 2) * 8 + 2000, np.arange(2) * 50 + 5000)
    header = {"receiver_line": b.flatten(), "receiver_number": a.flatten()}
    return data + noise, header
