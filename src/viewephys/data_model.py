import abc

import scipy.signal
from ibldsp import voltage


class AbstractDataModel(abc.ABC):
    @abc.abstractmethod
    def get_data(self, start_sample, end_sample, pp_step): ...

    @abc.abstractmethod
    def get_num_samples(self): ...

    @abc.abstractmethod
    def get_sampling_frequency(self): ...

    @abc.abstractmethod
    def get_recording_length(self): ...

    @abc.abstractmethod
    def get_file_path(self): ...

    @abc.abstractmethod
    def get_neuropixels_version(self): ...

    @abc.abstractmethod
    def get_num_channels(self): ...

    @abc.abstractmethod
    def get_saturation_adc(self): ...

    @abc.abstractmethod
    def get_geometry(self): ...


class SpikeGLXDataModel(AbstractDataModel):
    def __init__(self, sr):
        self.sr = sr

    def get_data(self, start_sample, end_sample, step, raw=None):
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

    def get_raw(self, start_sample, end_sample):
        """A temporary helper function for getting the raw data to match
        the old logic flow, without confusing recursion calling of get_data"""
        return self.sr[start_sample:end_sample, : self.sr.nc - self.sr.nsync].T

    def get_geometry(self):
        return self.sr.geometry

    def get_num_samples(self):
        return self.sr.ns

    def get_sampling_frequency(self):
        return self.sr.fs

    def get_recording_length(self):
        return self.sr.rl

    def get_file_path(self):
        return self.sr.file_bin

    def get_neuropixels_version(self):
        return self.sr.major_version

    def get_num_channels(self):
        return self.sr.nc

    def get_saturation_adc(self):
        return self.sr.range_volts[0] * 1e6
