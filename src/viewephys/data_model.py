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
        if not self._has_channel_locations(self.first_recording):
            geom = {"trace": np.arange(num_channels)}

        # Handle the second case, where channel locations are found
        # but the recording does not have a probe attached
        if (
            not self.first_recording.has_probe()
            or self.first_recording.get_probe().shank_ids is None
        ):
            positions = (
                self.first_recording.get_channel_locations()
            )  # (n_contacts, 2) in µm

            geom = {
                "trace": np.arange(num_channels),
                "x": positions[:, 0],
                "y": positions[:, 1],
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
        """
        Get the duration of the recording.

        SpikeInterface has a `get_times()` function
        that in theory can return a time array with
        non-constant sampling or sampling rate drift.
        We calculate from the times array, and use the
        fs as an approximation for the last sample, so
        the calculation convention is the same as other
        conditional paths.
        """
        times = self.first_recording.get_times()
        if times is not None:
            return (times[-1] - times[0]) + 1 / self.get_sampling_frequency()
        else:
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

    def perform_checks_on_recordings(self):  # noqa
        """ """
        if len(self.first_recording.get_probegroup().probes) > 1:
            raise NotImplementedError(
                "Multi-probe recordings are not supported yet. "
                "Please raise an issue on the viewephys GitHub "
                "if you would like to see this implemented."
            )

        first_has_locations = self._has_channel_locations(self.first_recording)
        has_probe = self.first_recording.has_probe()

        # Perform checks that all recordings are comparable and supported.
        for key in list(self.recordings_dict.keys())[1:]:
            rec = self.recordings_dict[key]

            if rec.get_num_segments() != 1:
                raise ValueError(
                    "Currently `viewephys` only supports 1 segment recordings. "
                    f"{key} has more than one."
                )

            rec_has_locations = self._has_channel_locations(rec)

            if rec_has_locations != first_has_locations:
                raise ValueError(
                    "The first recording "
                    f"{self.first_key} and recording {key} do not have the "
                    "same recording locations state. "
                    "Either all recordings must have contact locations or none "
                    "of them."
                )

            if rec.has_probe() != has_probe:
                raise ValueError(
                    "The first recording "
                    f"{self.first_key} and recording {key} do not have the "
                    "same probe attach state. "
                    "Either all recordings must have a probe attached or none "
                    "of them."
                )

        # Validate basic properties
        first_sampling_frequency = self.first_recording.get_sampling_frequency()
        first_num_samples = self.first_recording.get_num_samples(segment_index=0)
        first_gains = self.first_recording.get_channel_gains()
        first_offsets = self.first_recording.get_channel_offsets()

        if np.unique(first_gains).size != 1:
            raise ValueError(
                "All channels in the first recording "
                f"{self.first_key} must share the same gain."
            )
        if np.unique(first_offsets).size != 1:
            raise ValueError(
                "All channels in the first recording "
                f"{self.first_key} must share the same offset."
            )

        for key in list(self.recordings_dict.keys())[1:]:
            rec = self.recordings_dict[key]

            if rec.get_sampling_frequency() != first_sampling_frequency:
                raise ValueError(
                    "The sampling frequency for recording "
                    f"{key} ({rec.get_sampling_frequency()} Hz) does not "
                    "match the first recording "
                    f"{self.first_key} ({first_sampling_frequency} Hz)."
                )

            if rec.get_num_samples(segment_index=0) != first_num_samples:
                raise ValueError(
                    "The number of samples for recording "
                    f"{key} ({rec.get_num_samples(segment_index=0)}) does "
                    "not match the first recording "
                    f"{self.first_key} ({first_num_samples})."
                )

            if not np.array_equal(rec.get_channel_gains(), first_gains):
                raise ValueError(
                    "The channel gains for recording "
                    f"{key} do not match the first recording "
                    f"{self.first_key}."
                )

            if not np.array_equal(rec.get_channel_offsets(), first_offsets):
                raise ValueError(
                    "The channel offsets for recording "
                    f"{key} do not match the first recording "
                    f"{self.first_key}."
                )

            if first_has_locations and not np.array_equal(
                self.first_recording.get_channel_locations(),
                rec.get_channel_locations(),
            ):
                raise ValueError(
                    f"The channel locations for the first recording "
                    f"and {key} do not match."
                )

            if has_probe:
                rec_probe = rec.get_probe()
                first_probe = self.first_recording.get_probe()
                if not np.array_equal(
                    rec_probe.contact_positions, first_probe.contact_positions
                ):
                    raise ValueError(
                        "The contact locations on the probe do not match "
                        f"between recordings {self.first_key} and {key}"
                    )
                if not np.array_equal(rec_probe.shank_ids, first_probe.shank_ids):
                    raise ValueError(
                        "The shank IDs on the probe do not match between "
                        f"recordings {self.first_key} and {key}"
                    )

    def _has_channel_locations(self, recording) -> bool:
        try:
            recording.get_channel_locations()
            return True
        except Exception:
            return False
