from pathlib import Path

import numpy as np
from ibldsp import voltage
from spikeglx import Reader, _mock_spikeglx_file

from viewephys.data_model import SpikeGLXDataModel


class TestDataModel:
    def test_SpikeGLXDataModelr(self, tmp_path):
        """Simple pass-through checks for the wrapped SpikeGLX reader."""
        meta_path = tmp_path / "my_bin_file.bin"
        bin_path = Path(__file__).parent / "mock_data" / "meta_file.meta"

        file_info = _mock_spikeglx_file(
            meta_path,
            bin_path,
            100_000,
            384,
        )

        sr = Reader(file_info["bin_file"])

        model = SpikeGLXDataModel(sr)

        assert model.get_num_samples() == sr.ns
        assert model.get_sampling_frequency() == sr.fs
        assert model.get_recording_length() == sr.rl
        assert model.get_file_path() == sr.file_bin
        assert model.get_neuropixels_version() == sr.major_version
        assert model.get_num_channels() == sr.nc
        assert model.get_saturation_adc() == sr.range_volts[0] * 1e6
        assert model.get_header() is sr.geometry

        # We can pass a raw data file, in which case no slicing occurs
        random_raw = np.random.randn(sr.nc - sr.nsync, 32)
        assert np.array_equal(model.get_data(None, None, "raw", random_raw), random_raw)

        # otherwise we use the raw data from file
        assert np.array_equal(
            model.get_data(10, 20, "raw"), sr[10:20, : sr.nc - sr.nsync].T
        )

        # Similarly for destripe
        fcn_destripe = voltage.destripe_lfp if sr.type == "lf" else voltage.destripe

        expected = fcn_destripe(
            x=random_raw,
            fs=sr.fs,
            channel_labels=False,
            h=sr.geometry,
            neuropixel_version=sr.major_version,
        )
        np.testing.assert_allclose(
            model.get_data(None, None, "destripe", random_raw),
            expected,
        )

        raw_from_file = sr[10:42, : sr.nc - sr.nsync].T
        expected = fcn_destripe(
            x=raw_from_file,
            fs=sr.fs,
            channel_labels=False,
            h=sr.geometry,
            neuropixel_version=sr.major_version,
        )
        np.testing.assert_allclose(
            model.get_data(10, 42, "destripe"),
            expected,
        )

        # Filtering alone is not tested.

    def test_spikeinterface_no_channel_locs(self):
        pass

    def test_spikeinterface_no_probe(self):
        pass

    def test_spikeinterface_with_prbe(self):
        pass
