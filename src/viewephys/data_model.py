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
    def get_probe_information(self): ...

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
    def get_probe_information(self):
        return f"Neuropixels v{self.sr.major_version}"

    @override
    def get_num_channels(self) -> int:
        """Total channel count including sync channels."""
        return self.sr.nc

    @override
    def get_saturation_adc(self) -> float:
        """ADC saturation level in µV."""
        return self.sr.range_volts[0] * 1e6


class SpikeInterfaceDataModel(AbstractDataModel):
    def __init__(self, recordings_dict):
        self.recordings_dict = recordings_dict

        self.first_recording = recordings_dict[(next(iter(recordings_dict)))]

        # Perform checks that all recordings are comparable and supported.
        for rec in recordings_dict.values():
            if rec.get_num_segments() != 1:
                raise ValueError(
                    "Currently `viewephys` only supports 1 segment recordings."
                )

            if not rec.has_probe():
                raise ValueError("All passed recordings must have a probe attached.")

        first_probe = self.first_recording.get_probe()
        first_sampling_frequency = self.first_recording.get_sampling_frequency()
        first_num_samples = self.first_recording.get_num_samples(segment_index=0)
        first_gains = self.first_recording.get_channel_gains()
        first_offsets = self.first_recording.get_channel_offsets()

        if np.unique(first_gains).size != 1:
            raise ValueError(
                "All channels in the first recording must share the same gain."
            )
        if np.unique(first_offsets).size != 1:
            raise ValueError(
                "All channels in the first recording must share the same offset."
            )

        for key in list(self.recordings_dict.keys())[1:]:
            rec = self.recordings_dict[key]
            assert rec.get_sampling_frequency() == first_sampling_frequency
            assert rec.get_num_samples(segment_index=0) == first_num_samples
            assert np.array_equal(rec.get_channel_gains(), first_gains)
            assert np.array_equal(rec.get_channel_offsets(), first_offsets)
            assert rec.get_probe().to_dict() == first_probe.to_dict()

    def get_data(self, start_sample, end_sample, step, raw=None):
        assert step in self.recordings_dict, (
            "somehow the step names have become disconnected"
        )
        return (
            self.recordings_dict[step]
            .get_traces(
                start_frame=start_sample,
                end_frame=end_sample,
                return_in_uV=True,
                segment_index=0,
            )
            .T
        )

    # -------------------------------------------------------------------------
    # Implemented by GitHub Copilot (Claude Sonnet 4.6) — May 2026
    # -------------------------------------------------------------------------

    def get_raw(self, start_sample, end_sample):
        # SpikeInterface recordings carry their own preprocessing chain; the
        # shared raw-read cache used by SpikeGLXDataModel is not needed here.
        # Returning None is safe: get_data() ignores the `raw` argument for
        # this model.
        return None

    def get_geometry(self):
        breakpoint()
        probe = self.first_recording.get_probe()
        positions = probe.contact_positions  # (n_contacts, 2) in µm
        nc = self.first_recording.get_num_channels()
        return {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "trace": np.arange(nc),
        }

    def get_num_samples(self):
        return self.first_recording.get_num_samples(segment_index=0)

    def get_sampling_frequency(self):
        return self.first_recording.get_sampling_frequency()

    def get_recording_length(self):
        return self.get_num_samples() / self.get_sampling_frequency()

    def get_file_path(self):
        # SpikeInterface recordings are not necessarily file-backed.
        return None

    def get_probe_information(self):
        probe = self.first_recording.get_probe()
        a = probe.annotations
        parts = [a.get("manufacturer"), a.get("model_name"), a.get("serial_number")]
        return ", ".join(p for p in parts if p) or None

    def get_num_channels(self):
        return self.first_recording.get_num_channels()

    def get_saturation_adc(self):
        # ADC saturation range is hardware-specific and not available through
        # the SpikeInterface API.
        return None
