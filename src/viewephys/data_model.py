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
    def get_file_path(self) -> Path | None: ...

    @abc.abstractmethod
    def get_probe_information(self) -> str | None: ...

    @abc.abstractmethod
    def get_num_channels(self) -> int: ...

    @abc.abstractmethod
    def get_saturation_adc(self) -> float | None: ...

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
    def get_probe_information(self) -> str:
        return f"Neuropixels v{self.sr.major_version}"

    @override
    def get_num_channels(self) -> int:
        """Total channel count including sync channels."""
        return self.sr.nc

    @override
    def get_saturation_adc(self) -> float | None:
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

            # TODO: check for probegroup, not sure we can support at this stage

        first_contact_positions = self.first_recording.get_probe().contact_positions
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
            assert np.array_equal(
                rec.get_probe().contact_positions, first_contact_positions
            )

    @override
    def get_data(
        self,
        start_sample: int,
        end_sample: int,
        step: str,
        raw: np.ndarray | None = None,
    ) -> np.ndarray:
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

    def get_raw(self, start_sample: int, end_sample: int) -> None:
        """SpikeInterface recordings carry their own preprocessing chain"""
        return None

    @override
    def get_header(self) -> dict:
        probe = self.first_recording.get_probe()
        positions = probe.contact_positions  # (n_contacts, 2) in µm
        num_channels = self.first_recording.get_num_channels()

        _, row = np.unique(positions[:, 1], return_inverse=True)

        # col is computed per-shank so shank-relative x values are ranked correctly
        col = np.zeros(num_channels, dtype=float)
        for shank_id in np.unique(probe.shank_ids):
            mask = probe.shank_ids == shank_id
            _, col[mask] = np.unique(positions[mask, 0], return_inverse=True)

        geom = {
            "trace": np.arange(num_channels),
            "shank": probe.shank_ids.astype(int),
            "x": positions[:, 0],
            "y": positions[:, 1],
            # TODO: we have no flag yet. We can look for it on the SI recording
            "col": col,
            "row": row.astype(int),
        }

        if (
            sample_shift := self.first_recording.get_property("inter_sample_shift")
        ) is not None:
            geom["sample_shift"] = np.asarray(sample_shift, dtype=float)

        if (adc := probe.contact_annotations.get("adc_group")) is not None:
            geom["adc"] = np.asarray(adc, dtype=int)

        if probe.device_channel_indices is not None:
            geom["ind"] = probe.device_channel_indices.astype(int)

        return geom

    @override
    def get_num_samples(self) -> int:
        return self.first_recording.get_num_samples(segment_index=0)

    @override
    def get_sampling_frequency(self) -> float:
        return self.first_recording.get_sampling_frequency()

    @override
    def get_recording_length(self) -> float:
        return self.get_num_samples() / self.get_sampling_frequency()

    @override
    def get_file_path(self) -> None:
        # SpikeInterface recordings are not necessarily file-backed.
        return None

    @override
    def get_probe_information(self) -> str | None:
        probe = self.first_recording.get_probe()
        a = probe.annotations
        parts = [a.get("manufacturer"), a.get("model_name"), a.get("serial_number")]
        return ", ".join(p for p in parts if p) or None

    @override
    def get_num_channels(self) -> int:
        return self.first_recording.get_num_channels()

    @override
    def get_saturation_adc(self) -> None:
        # ADC saturation range is hardware-specific and not available through
        # the SpikeInterface API.
        return None
