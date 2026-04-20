import numpy as np


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
