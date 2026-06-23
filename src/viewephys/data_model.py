import abc
import warnings
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
    def get_times(self) -> np.ndarray: ...

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
        self._times: np.ndarray | None = None

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
            and do not want to repeatedly slice the raw data. Note `start_sample` and
            `end_sample` are unused when `raw` is not `None`.

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
    def get_times(self) -> np.ndarray:
        """Per-sample timestamps in seconds (starting at 0)."""
        if self._times is None:
            self._times = np.arange(self.sr.ns) / self.sr.fs
        return self._times

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


# TODO: check for probegroup, not sure we can support at this stage


class SpikeInterfaceDataModel(AbstractDataModel):
    def __init__(self, recordings_dict):
        self.recordings_dict = recordings_dict
        self._times: np.ndarray | None = None

        self.first_recording = recordings_dict[(next(iter(recordings_dict)))]
        self.first_key = list(recordings_dict.keys())[0]

        self.perform_checks_on_recordings()

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
        num_channels = self.first_recording.get_num_channels()

        # Handle the case where no channel locations can be found
        try:
            self.first_recording.get_channel_locations()
        except Exception:
            geom = {
                "trace": np.arange(num_channels),
                "ids": self.first_recording.get_channel_ids(),  # TODO: do not show this in trace dropdown
            }
            return geom

        # Handle the second case, where channel locations are found
        # but the recording does not have a probe attached
        if (
            not self.first_recording.has_probe()
            or self.first_recording.get_probe().shank_ids is None
            or self.first_recording.get_probe().shank_ids[0] == ""
        ):
            positions = (
                self.first_recording.get_channel_locations()
            )  # (n_contacts, 2) in µm

            geom = {
                "trace": np.arange(num_channels),
                "x": positions[:, 0],
                "y": positions[:, 1],
                "ids": self.first_recording.get_channel_ids(),  # TODO: do not show this in trace dropdown
            }
            return geom

        # Finally, the best case where a full probe is attached and we
        # can add shank and row/col information.
        probe = self.first_recording.get_probe()
        positions = probe.contact_positions  # (n_contacts, 2) in µm

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
            "ids": self.first_recording.get_channel_ids(),  # TODO: do not show this in trace dropdown
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
    def get_times(self) -> np.ndarray:
        """Per-sample timestamps in seconds from the underlying recording.

        SpikeInterface recordings may start at a non-zero time, so the times
        are taken from the recording rather than computed from the sampling
        frequency and sample count.
        """
        if self._times is None:
            self._times = self.first_recording.get_times(segment_index=0)
        return self._times

    @override
    def get_recording_length(self) -> float:
        return self.get_num_samples() / self.get_sampling_frequency()

    @override
    def get_file_path(self) -> None:
        # SpikeInterface recordings are not necessarily file-backed.
        return None

    @override
    def get_probe_information(self) -> str | None:
        if self.first_recording.has_probe():
            probe = self.first_recording.get_probe()
            a = probe.annotations
            parts = [a.get("manufacturer"), a.get("model_name"), a.get("serial_number")]
            return ", ".join(p for p in parts if p) or None
        else:
            return None

    @override
    def get_num_channels(self) -> int:
        return self.first_recording.get_num_channels()

    @override
    def get_saturation_adc(self) -> None:
        # ADC saturation range is hardware-specific and not available through
        # the SpikeInterface API.
        return None

    def perform_checks_on_recordings(self):
        """ """
        try:
            self.first_recording.get_channel_locations()
            first_has_locations = True
        except Exception:
            first_has_locations = False

        has_probe = self.first_recording.has_probe()

        # Perform checks that all recordings are comparable and supported.
        for key, rec in self.recordings_dict.items():
            if rec.get_num_segments() != 1:
                raise ValueError(
                    f"Currently `viewephys` only supports 1 segment recordings. {key} has more than one."
                )

            try:
                rec.get_channel_locations()
                rec_has_locations = True
            except Exception:
                rec_has_locations = False

            if rec_has_locations and not first_has_locations:
                raise ValueError(
                    f"The first recording does not have contact locations, "
                    f"but other recordings (e.g. {key}) do."
                )
            if first_has_locations and not rec_has_locations:
                warnings.warn(
                    f"The first recording {self.first_key} has contact locations, but {key} does not."
                )

            if rec.has_probe() and not has_probe:
                raise ValueError(
                    f"The first recording {self.first_key} does not have a probe attached, "
                    f"but other recordings (e.g. {key}) do."
                )

            if has_probe and not rec.has_probe():
                raise ValueError(
                    f"The first recording {self.first_key} has a probe attached, but {key} does not"
                )

        first_sampling_frequency = self.first_recording.get_sampling_frequency()
        first_num_samples = self.first_recording.get_num_samples(segment_index=0)
        first_gains = self.first_recording.get_channel_gains()
        first_offsets = self.first_recording.get_channel_offsets()

        if np.unique(first_gains).size != 1:
            raise ValueError(
                f"All channels in the first recording {self.first_key} must share the same gain."
            )
        if np.unique(first_offsets).size != 1:
            raise ValueError(
                f"All channels in the first recording {self.first_key} must share the same offset."
            )

        for key in list(self.recordings_dict.keys())[1:]:
            rec = self.recordings_dict[key]

            if rec.get_sampling_frequency() != first_sampling_frequency:
                raise ValueError(
                    f"The sampling frequency for recording {key} "
                    f"({rec.get_sampling_frequency()} Hz) does not match the first recording "
                    f"{self.first_key} ({first_sampling_frequency} Hz)."
                )

            if rec.get_num_samples(segment_index=0) != first_num_samples:
                raise ValueError(
                    f"The number of samples for recording {key} "
                    f"({rec.get_num_samples(segment_index=0)}) does not match the first recording "
                    f"{self.first_key} ({first_num_samples})."
                )

            if not np.array_equal(rec.get_channel_gains(), first_gains):
                raise ValueError(
                    f"The channel gains for recording {key} do not match the first recording "
                    f"{self.first_key}."
                )

            if not np.array_equal(rec.get_channel_offsets(), first_offsets):
                raise ValueError(
                    f"The channel offsets for recording {key} do not match the first recording "
                    f"{self.first_key}."
                )

            if first_has_locations:
                try:
                    rec_locs = rec.get_channel_locations()
                except Exception:
                    rec_locs = None
                if rec_locs is not None:
                    if not np.array_equal(
                        self.first_recording.get_channel_locations(),
                        rec.get_channel_locations(),
                    ):
                        raise ValueError(
                            f"The channel locations for the first recording and"
                            f"{key} do not match."
                        )

            if has_probe:
                if rec.has_probe():
                    rec_probe = rec.get_probe()
                    first_probe = self.first_recording.get_probe()
                    if not np.array_equal(
                        rec_probe.contact_positions, first_probe.contact_positions
                    ):
                        raise ValueError(
                            f"The contact locations on the probe do not match "
                            f"between recordings {self.first_key} and {key}"
                        )
                    if not np.array_equal(rec_probe.shank_ids, first_probe.shank_ids):
                        raise ValueError(
                            f"The shank IDs on the probe do not "
                            f"match between recordings {self.first_key} and {key}"
                        )
