"""
Noise (RMS) and DC offset estimation on snippets.

Per-channel RMS of the AP band before and after the standard preprocessing
steps of the IBL pipeline (HP Butterworth filter and destriping), as exposed
by :meth:`viewephys.data_model.SpikeGLXDataModel.get_data`, and per-channel
DC offset of the raw signal.
"""

import numpy as np
import pandas as pd

from viewephys.data_model import SpikeGLXDataModel

DEFAULT_STEPS = ("raw", "butterworth", "destripe")
DEFAULT_FRACTIONS = (0.05, 0.3, 0.55, 0.8)


def array_rms(
    data: np.ndarray,
    step: str,
    start_sample: int,
    end_sample: int,
    fs: float,
) -> pd.DataFrame:
    """
    Compute per-channel RMS of one already-loaded snippet array.

    Parameters
    ----------
    data
        Snippet array of shape ``(n_channels, n_samples)``, in volts.
    step
        Name of the preprocessing step the array corresponds to.
    start_sample
        First sample of the snippet.
    end_sample
        Last sample (exclusive) of the snippet.
    fs
        Sampling frequency in Hz.

    Returns
    -------
    pd.DataFrame
        One row per channel with columns ``step``, ``channel``, ``rms_volts``,
        ``start_sample``, ``end_sample`` and ``sampling_frequency``.
    """
    rms = np.sqrt(np.mean(np.square(data, dtype=np.float64), axis=1))
    return pd.DataFrame(
        {
            "step": step,
            "channel": np.arange(data.shape[0]),
            "rms_volts": rms,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "sampling_frequency": fs,
        }
    )


def snippet_rms(
    data_model: SpikeGLXDataModel,
    start_sample: int,
    end_sample: int,
    steps: tuple[str, ...] = DEFAULT_STEPS,
) -> pd.DataFrame:
    """
    Compute per-channel RMS on a single snippet for each preprocessing step.

    The raw snippet is read once and reused for all steps, so the preprocessing
    applied is exactly the one of the data model (e.g. ``"butterworth"`` HP
    filter and ``"destripe"`` for a SpikeGLX AP band).

    Parameters
    ----------
    data_model
        Data model wrapping the recording, e.g. ``SpikeGLXDataModel``.
    start_sample
        First sample of the snippet.
    end_sample
        Last sample (exclusive) of the snippet.
    steps
        Preprocessing steps to evaluate, among ``data_model.get_steps()``.

    Returns
    -------
    pd.DataFrame
        One row per (step, channel) with columns ``step``, ``channel``,
        ``rms_volts``, ``start_sample``, ``end_sample`` and
        ``sampling_frequency``.
    """
    raw = data_model.get_raw(start_sample, end_sample)
    fs = data_model.get_sampling_frequency()
    frames = [
        array_rms(
            data_model.get_data(start_sample, end_sample, step, raw=raw),
            step=step,
            start_sample=start_sample,
            end_sample=end_sample,
            fs=fs,
        )
        for step in steps
    ]
    return pd.concat(frames, ignore_index=True)


def recording_rms(
    data_model: SpikeGLXDataModel,
    snippet_sec: float = 3.0,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    steps: tuple[str, ...] = DEFAULT_STEPS,
) -> pd.DataFrame:
    """
    Compute per-channel RMS on multiple snippets spread across the recording.

    Parameters
    ----------
    data_model
        Data model wrapping the recording, e.g. ``SpikeGLXDataModel``.
    snippet_sec
        Duration of each snippet in seconds.
    fractions
        Positions of the snippet starts as fractions of the recording length.
    steps
        Preprocessing steps to evaluate, among ``data_model.get_steps()``.

    Returns
    -------
    pd.DataFrame
        Concatenation of the :func:`snippet_rms` frames, one row per
        (snippet, step, channel).
    """
    fs = data_model.get_sampling_frequency()
    ns = data_model.get_num_samples()
    n_samples = int(snippet_sec * fs)
    frames = []
    for fraction in fractions:
        start_sample = int(ns * fraction)
        end_sample = min(start_sample + n_samples, ns)
        frames.append(snippet_rms(data_model, start_sample, end_sample, steps))
    return pd.concat(frames, ignore_index=True)


def summarize_rms(rms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a per-channel RMS dataframe into a report table.

    Parameters
    ----------
    rms_df
        Dataframe as returned by :func:`recording_rms`.

    Returns
    -------
    pd.DataFrame
        One row per preprocessing step with columns ``step``, ``n_snippets``,
        ``n_channels``, ``mean_rms_volts``, ``se_rms_volts`` (standard error over
        all channel x snippet values), ``median_rms_volts``, ``mad_rms_volts``
        (robust spread, 1.4826 * MAD) and ``mean_rms_uv``.
    """
    rows = []
    for step, group in rms_df.groupby("step", sort=False):
        rms = group["rms_volts"].to_numpy(dtype=np.float64)
        median = np.median(rms)
        rows.append(
            {
                "step": step,
                "n_snippets": group["start_sample"].nunique(),
                "n_channels": group["channel"].nunique(),
                "mean_rms_volts": rms.mean(),
                "se_rms_volts": rms.std(ddof=1) / np.sqrt(rms.size),
                "median_rms_volts": median,
                "mad_rms_volts": 1.4826 * np.median(np.abs(rms - median)),
                "mean_rms_uv": rms.mean() * 1e6,
            }
        )
    return pd.DataFrame(rows)


def array_dc_offset(
    data: np.ndarray,
    start_sample: int,
    end_sample: int,
    fs: float,
) -> pd.DataFrame:
    """
    Compute per-channel DC offset (mean) of one already-loaded raw snippet.

    Parameters
    ----------
    data
        Raw snippet array of shape ``(n_channels, n_samples)``, in volts.
    start_sample
        First sample of the snippet.
    end_sample
        Last sample (exclusive) of the snippet.
    fs
        Sampling frequency in Hz.

    Returns
    -------
    pd.DataFrame
        One row per channel with columns ``channel``, ``dc_volts``,
        ``start_sample``, ``end_sample`` and ``sampling_frequency``.
    """
    return pd.DataFrame(
        {
            "channel": np.arange(data.shape[0]),
            "dc_volts": np.mean(data, axis=1, dtype=np.float64),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "sampling_frequency": fs,
        }
    )


def snippet_dc_offset(
    data_model: SpikeGLXDataModel,
    start_sample: int,
    end_sample: int,
) -> pd.DataFrame:
    """
    Compute per-channel DC offset of the raw signal on a single snippet.

    Parameters
    ----------
    data_model
        Data model wrapping the recording, e.g. ``SpikeGLXDataModel``.
    start_sample
        First sample of the snippet.
    end_sample
        Last sample (exclusive) of the snippet.

    Returns
    -------
    pd.DataFrame
        One row per channel with columns ``channel``, ``dc_volts``,
        ``start_sample``, ``end_sample`` and ``sampling_frequency``.
    """
    raw = data_model.get_raw(start_sample, end_sample)
    return array_dc_offset(
        raw,
        start_sample=start_sample,
        end_sample=end_sample,
        fs=data_model.get_sampling_frequency(),
    )


def recording_dc_offset(
    data_model: SpikeGLXDataModel,
    snippet_sec: float = 3.0,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> pd.DataFrame:
    """
    Compute per-channel DC offset of the raw signal on multiple snippets.

    Parameters
    ----------
    data_model
        Data model wrapping the recording, e.g. ``SpikeGLXDataModel``.
    snippet_sec
        Duration of each snippet in seconds.
    fractions
        Positions of the snippet starts as fractions of the recording length.

    Returns
    -------
    pd.DataFrame
        Concatenation of the :func:`snippet_dc_offset` frames, one row per
        (snippet, channel).
    """
    fs = data_model.get_sampling_frequency()
    ns = data_model.get_num_samples()
    n_samples = int(snippet_sec * fs)
    frames = []
    for fraction in fractions:
        start_sample = int(ns * fraction)
        end_sample = min(start_sample + n_samples, ns)
        frames.append(snippet_dc_offset(data_model, start_sample, end_sample))
    return pd.concat(frames, ignore_index=True)


def summarize_dc_offset(dc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a per-channel DC offset dataframe into a report table.

    Parameters
    ----------
    dc_df
        Dataframe as returned by :func:`recording_dc_offset`.

    Returns
    -------
    pd.DataFrame
        One row per snippet (``start_sample``) with columns ``n_channels``,
        ``median_dc_volts``, ``min_dc_volts``, ``max_dc_volts``,
        ``mad_dc_volts`` (robust spread, 1.4826 * MAD) and ``median_dc_uv``.
    """
    rows = []
    for start_sample, group in dc_df.groupby("start_sample", sort=False):
        dc = group["dc_volts"].to_numpy(dtype=np.float64)
        median = np.median(dc)
        rows.append(
            {
                "start_sample": start_sample,
                "n_channels": group["channel"].nunique(),
                "median_dc_volts": median,
                "min_dc_volts": dc.min(),
                "max_dc_volts": dc.max(),
                "mad_dc_volts": 1.4826 * np.median(np.abs(dc - median)),
                "median_dc_uv": median * 1e6,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    import spikeglx

    sr = spikeglx.Reader(sys.argv[1])
    try:
        model = SpikeGLXDataModel(sr)
        print(summarize_rms(recording_rms(model)).to_string(index=False))
        print(summarize_dc_offset(recording_dc_offset(model)).to_string(index=False))
    finally:
        sr.close()
