import abc
from pathlib import Path

import numpy as np
import scipy.signal
from ibldsp import voltage
from typing_extensions import override


class AbstractDataModel(abc.ABC):
    """
    Abstract class to interface with data loaded from different backends.
    """

    @abc.abstractmethod
    def get_data(
        self,
        start_sample: int,
        end_sample: int,
        pp_step: str,
        raw: np.ndarray | None = None,
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def get_num_samples(self) -> int: ...

    @abc.abstractmethod
    def get_sampling_frequency(self) -> float: ...

    @abc.abstractmethod
    def get_recording_length(self) -> float: ...

    @abc.abstractmethod
    def get_file_path(self) -> Path: ...

    @abc.abstractmethod
    def get_neuropixels_version(self) -> int: ...

    @abc.abstractmethod
    def get_num_channels(self) -> int: ...

    @abc.abstractmethod
    def get_saturation_adc(self) -> float: ...

    @abc.abstractmethod
    def get_header(self) -> dict: ...


class SpikeGLXDataModel(AbstractDataModel):
    """Data model wrapping ``spikeglx.Reader``."""

    def __init__(self, sr) -> None:
        self.sr = sr

    @override
    def get_data(
        self,
        start_sample: int,
        end_sample: int,
        step: str,
        raw: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return the raw or preprocessed data.

        Here the steps are fixed by the existing IBL preprocessing pipeline.

        Parameters
        ----------
        start_sample
            First sample of the data to return
        end_sample
            Last sample (exclusive) of the data to return
        step
            The preprocessing step: one of ``"raw"``, ``"destripe"``,
            ``"butterworth"``, or ``"broadband"``.
        raw
            Option to pass the raw data. This can be used when calling this many times
            and do not want to repeatedly slice the raw data.

        Returns
        -------
        np.ndarray of shape ``(n_channels, n_samples)``.
        """
        if raw is None:
            raw = self.get_raw(start_sample, end_sample)

        match step:
            case "raw":
                data = raw
            case "destripe":
                fcn_destripe = (
                    voltage.destripe_lfp if self.sr.type == "lf" else voltage.destripe
                )

                data = fcn_destripe(
                    x=raw,
                    fs=self.sr.fs,
                    channel_labels=False,
                    h=self.sr.geometry,
                    neuropixel_version=self.sr.major_version,
                )

            case "butterworth":
                cutoff = 3 if self.sr.type == "lf" else 300
                butter_kwargs = {
                    "N": 3,
                    "Wn": cutoff / self.sr.fs * 2,
                    "btype": "highpass",
                }

                sos = scipy.signal.butter(**butter_kwargs, output="sos")
                data = scipy.signal.sosfiltfilt(sos, raw)

            case "broadband":
                last = start_sample + int(self.sr.fs * 3)
                raw = self.get_raw(start_sample, last)
                butter_kwargs = {
                    "N": 3,
                    "Wn": 2 / self.sr.fs * 2,
                    "btype": "highpass",
                }
                sos = scipy.signal.butter(**butter_kwargs, output="sos")
                data = scipy.signal.sosfiltfilt(sos, raw)

        return data

    def get_raw(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Return raw data as ``(n_channels, n_samples)``, sync channels excluded."""
        return self.sr[start_sample:end_sample, : self.sr.nc - self.sr.nsync].T

    @override
    def get_header(self) -> dict:
        """
        Returns the header holding information on the probe,
        including x, y positions, shank mask, adc index, sample offset.
        """
        return self.sr.geometry

    @override
    def get_num_samples(self) -> int:
        """Number of samples in the recording."""
        return self.sr.ns

    @override
    def get_sampling_frequency(self) -> float:
        """Sampling frequency (Hz) of the recording."""
        return self.sr.fs

    @override
    def get_recording_length(self) -> float:
        """Recording length in seconds."""
        return self.sr.rl

    @override
    def get_file_path(self) -> Path:
        """Path to the raw .bin file."""
        return self.sr.file_bin

    @override
    def get_neuropixels_version(self) -> int:
        """Neuropixels probe major version."""
        return self.sr.major_version

    @override
    def get_num_channels(self) -> int:
        """Total channel count including sync channels."""
        return self.sr.nc

    @override
    def get_saturation_adc(self) -> float:
        """ADC saturation level in µV."""
        return self.sr.range_volts[0] * 1e6
