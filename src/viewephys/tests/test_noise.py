"""Unit tests for noise module."""

import numpy as np
import pytest

from viewephys.noise import (
    array_dc_offset,
    array_rms,
    array_saturation,
)

FS = 30000.0

# RMS tests


def test_array_rms_constant_and_sine():
    """
    A constant has RMS equal to its own value,
    a sine of amplitude X has X / sqrt(2).
    """
    n = 30000
    t = np.arange(n) / FS
    data = np.vstack(
        [
            np.full(n, 2e-5),  # constant 20 uV
            4e-5 * np.sin(2 * np.pi * 100 * t),  # 40 uV amp
            np.zeros(n),
        ]
    )
    df = array_rms(data, step="raw", start_sample=0, end_sample=n, fs=FS)

    assert list(df["channel"]) == [0, 1, 2]
    assert df["step"].unique() == ["raw"]
    np.testing.assert_allclose(
        df["rms_volts"], [2e-5, 4e-5 / np.sqrt(2), 0.0], rtol=1e-6, atol=1e-12
    )
    assert (df["start_sample"] == 0).all() and (df["end_sample"] == n).all()
    assert (df["sampling_frequency"] == FS).all()


# DC offset tests


def test_array_dc_offset_recovers_per_channel_mean():
    """The per-channel mean of zero-mean noise plus an offset is the offset."""
    rng = np.random.default_rng(0)
    n = 100000
    offsets = np.array([0.0, 1e-4, -2e-4])
    noise = rng.standard_normal((3, n)) * 1e-6
    noise -= noise.mean(axis=1, keepdims=True)  # exactly zero-mean
    df = array_dc_offset(noise + offsets[:, None], start_sample=0, end_sample=n, fs=FS)

    np.testing.assert_allclose(df["dc_volts"], offsets, atol=1e-12)


# Saturation tests


def test_array_saturation_counts_clipped_blocks():
    """Samples at or beyond the per-channel limit are counted, others are not."""
    n = 1000
    data = np.zeros((4, n))
    data[1, 100:150] = 0.6  # 50 samples at the positive limit
    data[2, 0:10] = -0.6  # 10 samples at the negative limit
    data[3, 500] = 0.59  # just under: not saturated
    df = array_saturation(data, np.full(4, 0.6), start_sample=0, end_sample=n, fs=FS)

    assert df["n_saturated"].tolist() == [0, 50, 10, 0]
    np.testing.assert_allclose(df["fraction_saturated"], [0.0, 0.05, 0.01, 0.0])
    assert (df["start_sample"] == 0).all() and (df["end_sample"] == n).all()
    assert (df["sampling_frequency"] == FS).all()


def test_array_saturation_catches_both_rails_one_lsb_under_full_scale():
    """NP1 integers run -512 to 511: with the threshold one LSB under full scale
    both rails count; at full scale the positive rail is missed."""
    lsb = 2.34375e-06  # volts per bit on an NP1 AP channel
    full_scale = 512 * lsb
    data = np.zeros((2, 100))
    data[0, :5] = 511 * lsb  # positive rail
    data[1, :7] = -512 * lsb  # negative rail

    at_full_scale = array_saturation(data, np.full(2, full_scale), 0, 100, FS)
    assert at_full_scale["n_saturated"].tolist() == [0, 7]

    one_lsb_under = array_saturation(data, np.full(2, full_scale - lsb), 0, 100, FS)
    assert one_lsb_under["n_saturated"].tolist() == [5, 7]


def test_array_saturation_rejects_mismatched_range_shape():
    """Passing sr.range_volts with the sync channel still in must not pass silently."""
    data = np.zeros((4, 10))
    with pytest.raises(ValueError):
        array_saturation(data, np.full(5, 0.6), 0, 10, FS)
