from pathlib import Path

import numpy as np
import probeinterface as pi
import pytest
import spikeinterface.core as si_core
import spikeinterface.preprocessing as si_prepro
from ibldsp import voltage
from numpy.testing import assert_equal as np_assert_equal
from spikeglx import Reader, _mock_spikeglx_file

from viewephys.data_model import (
    LFPackDataModel,
    SpikeGLXDataModel,
    SpikeInterfaceDataModel,
)
from viewephys.tests.test_viewer_helpers import build_lfpack_h5


class TestSpikeGLXDataModel:
    def test_spikeglx_mode(self, tmp_path):
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
        assert model.get_probe_information() == "Neuropixels v2.4"
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
        assert model.get_steps() == ["raw", "butterworth", "destripe", "broadband"]


class TestLFPackDataModel:
    def test_single_recording(self, tmp_path):
        lfpack = pytest.importorskip("lfpack")
        h5 = build_lfpack_h5(tmp_path / "lf.h5", "uuid-aaaa", nc=64, ns=6000)

        reader = lfpack.LFPackReader(h5)
        model = LFPackDataModel(reader)

        # A packed file exposes a single, already-denoised signal.
        assert model.get_steps() == ["raw"]
        assert model.get_num_channels() == 64
        assert model.get_sampling_frequency() == 250.0
        assert model.get_num_samples() == 6000
        assert model.get_recording_length() == pytest.approx(6000 / 250.0)
        assert model.get_probe_information() == "Neuropixels LFP (lfpack)"
        # No ADC saturation / voltage range in compressed files.
        assert model.get_saturation_adc() is None
        assert model.get_file_path() == h5

        # Probe geometry is available as the trace header.
        header = model.get_header()
        assert "x" in header and "y" in header
        assert header["x"].size == 64

        # Data comes back channel-first, in volts.
        data = model.get_data(0, 500, "raw")
        assert data.shape == (64, 500)
        assert data.dtype == np.float32

        # get_data passes a pre-fetched raw buffer straight through.
        raw = model.get_raw(0, 100)
        assert raw.shape == (64, 100)
        assert np.array_equal(model.get_data(0, 100, "raw", raw=raw), raw)

    def test_no_annotations_has_no_brain_regions(self, tmp_path):
        lfpack = pytest.importorskip("lfpack")
        h5 = build_lfpack_h5(tmp_path / "lf.h5", "uuid-aaaa", nc=32, ns=4096)
        model = LFPackDataModel(lfpack.LFPackReader(h5))

        header = model.get_header()
        assert "atlas_id" not in header
        assert model.get_brain_regions() is None

    def test_channel_annotations_and_brain_regions(self, tmp_path):
        lfpack = pytest.importorskip("lfpack")
        pytest.importorskip("iblatlas")
        h5 = build_lfpack_h5(
            tmp_path / "lf.h5", "uuid-aaaa", nc=32, ns=4096, annotate=True
        )
        model = LFPackDataModel(lfpack.LFPackReader(h5))

        # Brain-region annotations flow into the channel/header table.
        header = model.get_header()
        assert "atlas_id" in header and "acronym" in header
        assert header["atlas_id"].shape == (32,)
        # acronym is an ndarray so it indexes like the numeric header fields.
        assert isinstance(header["acronym"], np.ndarray)

        # A single BrainRegions instance is built and reused.
        br = model.get_brain_regions()
        assert br is not None
        assert hasattr(br, "id") and hasattr(br, "rgb")
        assert model.get_brain_regions() is br

    def test_recordings_listing_multi(self, tmp_path):
        lfpack = pytest.importorskip("lfpack")
        f1 = build_lfpack_h5(tmp_path / "a.h5", "uuid-aaaa", nc=32, ns=4096)
        f2 = build_lfpack_h5(tmp_path / "b.h5", "uuid-bbbb", nc=32, ns=4096)
        multi = tmp_path / "multi.h5"
        lfpack.merge_h5([f1, f2], multi)

        recordings = lfpack.LFPackReader.recordings(multi)
        assert set(recordings) == {"uuid-aaaa", "uuid-bbbb"}

        # Each recording opens into an independent, valid data model.
        for rec in recordings:
            model = LFPackDataModel(lfpack.LFPackReader(multi, recording=rec))
            assert model.get_num_channels() == 32
            assert model.get_num_samples() == 4096


class TestSpikeInterfaceDataModel:
    def test_spikeinterface_no_channel_locs(self):
        """
        Test spikeinterface model is correct in the case no channel locations
        are attached to the probe. In this case, the header is minimal.
        """
        rec = si_core.generate_recording(
            num_channels=4,
            sampling_frequency=30000.0,
            durations=[0.1],
            set_probe=False,
            seed=0,
        )
        filtered = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)

        model = SpikeInterfaceDataModel({"raw": rec, "filtered": filtered})
        # add wrong step and check kit
        # check all the errors in checkthing
        np_assert_equal(
            model.get_header(), {"trace": np.arange(model.get_num_channels())}
        )
        self.assert_all_spikeinterface_methods(
            rec,
            filtered,
            model,
            expected_duration=0.1,
        )

    def test_spikeinterface_no_probe(self):
        """
        Test spikeinterface model is correct in the case channel locations
        are attached to the probe. In this case, the header includes x, y information
        """
        rec = si_core.generate_recording(
            num_channels=5,
            sampling_frequency=40000.0,
            durations=[0.2],
            set_probe=False,
            seed=0,
        )
        channel_locs = np.array(
            [[0.0, 10.0], [1.0, 11.0], [0.0, 20.0], [1.0, 22.0], [0.0, 30.0]]
        )
        rec.set_channel_locations(channel_locs)

        filtered = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)

        model = SpikeInterfaceDataModel({"raw": rec, "filtered": filtered})
        np_assert_equal(
            model.get_header(),
            {
                "trace": np.arange(rec.get_num_channels()),
                "x": channel_locs[:, 0],
                "y": channel_locs[:, 1],
            },
        )
        self.assert_all_spikeinterface_methods(
            rec, filtered, model, expected_duration=0.2
        )

    def test_spikeinterface_with_probe(self):
        """
        Test spikeinterface model is correct in the case a full probe is
        attached to the probe. In this case, the header includes many entries.
        """
        probe = pi.Probe(ndim=2)
        shanks = np.array([0, 1, 0, 1, 0])
        channel_locs = np.array(
            [[0.0, 10.0], [1.0, 11.0], [0.0, 20.0], [1.0, 22.0], [0.1, 22.0]],
        )
        sample_shifts = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        adc_group = np.array([10, 11, 12, 13, 14])

        probe.set_contacts(positions=channel_locs, shank_ids=shanks)
        # probe.set_shank_ids(shanks)
        probe.set_device_channel_indices(np.arange(5))
        probe.contact_annotations["adc_group"] = adc_group
        rec = si_core.generate_recording(
            num_channels=5,
            sampling_frequency=55000.0,
            durations=[0.3],
            set_probe=False,
            seed=0,
        )
        rec = rec.set_probe(probe, group_mode="by_probe")
        rec.set_property("inter_sample_shift", sample_shifts)

        filtered = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)

        model = SpikeInterfaceDataModel({"raw": rec, "filtered": filtered})
        np_assert_equal(
            model.get_header(),
            {
                "trace": np.arange(rec.get_num_channels()),
                "shank": shanks,
                "x": channel_locs[:, 0],
                "y": channel_locs[:, 1],
                "col": [0, 0, 0, 0, 1],
                "row": [0, 1, 2, 3, 3],
                "sample_shift": sample_shifts,
                "adc": adc_group,
                "ind": np.arange(5),
            },
        )
        self.assert_all_spikeinterface_methods(
            rec, filtered, model, expected_duration=0.3
        )

    def test_spikeinterface_with_probe_non_int_shank_ids(self):
        """
        Test spikeinterface model when the probe shank ids cannot be cast to
        int (e.g. "s1", "s2"). In this case the "shank" key is omitted from
        the header while all position-based entries are still present.
        """
        probe = pi.Probe(ndim=2)
        shanks = np.array(["s0", "s1", "s0", "s1", "s0"])
        channel_locs = np.array(
            [[0.0, 10.0], [1.0, 11.0], [0.0, 20.0], [1.0, 22.0], [0.1, 22.0]],
        )

        probe.set_contacts(positions=channel_locs, shank_ids=shanks)
        probe.set_device_channel_indices(np.arange(5))
        rec = si_core.generate_recording(
            num_channels=5,
            sampling_frequency=55000.0,
            durations=[0.3],
            set_probe=False,
            seed=0,
        )
        rec = rec.set_probe(probe, group_mode="by_probe")

        filtered = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)

        model = SpikeInterfaceDataModel({"raw": rec, "filtered": filtered})
        header = model.get_header()

        assert "shank" not in header
        np_assert_equal(
            header,
            {
                "trace": np.arange(rec.get_num_channels()),
                "x": channel_locs[:, 0],
                "y": channel_locs[:, 1],
                "col": [0, 0, 0, 0, 1],
                "row": [0, 1, 2, 3, 3],
                "ind": np.arange(5),
            },
        )
        self.assert_all_spikeinterface_methods(
            rec, filtered, model, expected_duration=0.3
        )

    def assert_all_spikeinterface_methods(
        self, rec, filtered, model, expected_duration
    ):
        """"""
        assert np.array_equal(
            rec.get_traces(start_frame=0, end_frame=5, return_in_uV=True).T,
            model.get_data(0, 5, "raw"),
        )
        assert np.array_equal(
            filtered.get_traces(start_frame=0, end_frame=5, return_in_uV=True).T,
            model.get_data(0, 5, "filtered"),
        )
        assert model.get_raw(0, 5) is None
        assert rec.get_num_samples() == model.get_num_samples()
        assert rec.get_sampling_frequency() == model.get_sampling_frequency()
        assert model.get_file_path() is None
        assert rec.get_num_channels() == model.get_num_channels()
        assert model.get_probe_information() is None
        assert model.get_saturation_adc() is None

        duration = model.get_recording_length()
        assert np.isclose(duration, expected_duration, rtol=0, atol=1e-8)

    def test_recording_checks(self):
        # Start from one plain recording with no attached probe or contact
        # locations; every other fixture in this test is derived from this
        # baseline so the setup stays compact and easy to reason about.
        base = si_core.generate_recording(
            num_channels=4,
            sampling_frequency=1000.0,
            durations=[0.2],
            set_probe=False,
            seed=0,
        )
        other = base.clone()

        # A matching pair of recordings should pass the checks and expose the
        # same basic metadata as the first recording.
        model = SpikeInterfaceDataModel({"raw": base, "processed": other})

        assert model.get_num_channels() == base.get_num_channels()
        assert model.get_sampling_frequency() == base.get_sampling_frequency()
        assert model.get_num_samples() == base.get_num_samples(segment_index=0)
        assert model.get_recording_length() == pytest.approx(0.2)

        mismatched_fs = si_core.generate_recording(
            num_channels=4,
            sampling_frequency=2000.0,
            durations=[0.2],
            set_probe=False,
            seed=2,
        )

        # Different sampling frequencies should be rejected because the
        # recordings are not compatible preprocessing views of the same signal.
        with pytest.raises(ValueError, match="sampling frequency"):
            SpikeInterfaceDataModel({"raw": base, "processed": mismatched_fs})

        # One recording with contact locations and another without should fail
        # the recording-state consistency check.
        loc_rec = base.clone()
        loc_rec.set_channel_locations(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        with pytest.raises(ValueError, match="recording locations state"):
            SpikeInterfaceDataModel({"raw": base, "processed": loc_rec})

        loc_base = base.clone()
        loc_base.set_channel_locations(
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )
        loc_other = base.clone()
        loc_other.set_channel_locations(
            np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        )

        # Even when both recordings have locations, the exact coordinates must
        # match or the model will reject the pair.
        with pytest.raises(ValueError, match="channel locations"):
            SpikeInterfaceDataModel({"raw": loc_base, "processed": loc_other})

        # A probe-attached recording and a plain recording should not be mixed.
        # The smallest valid probe is enough here because this branch only
        # checks the attached-vs-unattached state, not the shank IDs.
        locs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        base_with_locs = base.clone()
        base_with_locs.set_channel_locations(locs)

        probe_rec = base.clone()
        probe_rec.set_channel_locations(locs)
        probe = pi.Probe(ndim=2)
        probe.set_contacts(positions=locs)
        probe.set_device_channel_indices(np.arange(4))
        probe_rec = probe_rec.set_probe(probe, group_mode="by_probe")

        with pytest.raises(ValueError, match="probe attach state"):
            SpikeInterfaceDataModel({"raw": base_with_locs, "processed": probe_rec})

        # A mismatched probe definition should also fail, especially when the
        # shank IDs differ between otherwise compatible recordings.
        other_probe = pi.Probe(ndim=2)
        other_probe.set_contacts(positions=locs)
        other_probe.set_shank_ids(np.array([0, 0, 0, 0]))
        other_probe.set_device_channel_indices(np.arange(4))
        probe_mismatch = base.clone()
        probe_mismatch.set_channel_locations(locs)
        probe_mismatch = probe_mismatch.set_probe(other_probe, group_mode="by_probe")

        with pytest.raises(ValueError, match="shank IDs"):
            SpikeInterfaceDataModel({"raw": probe_rec, "processed": probe_mismatch})

    def test_get_steps_returns_recording_dict_keys(self):
        """get_steps() must reflect the keys the caller passed in."""
        rec = si_core.generate_recording(
            num_channels=4,
            sampling_frequency=30000.0,
            durations=[0.1],
            set_probe=False,
            seed=0,
        )
        filtered = si_prepro.bandpass_filter(rec, freq_min=300, freq_max=6000)
        model = SpikeInterfaceDataModel({"raw": rec, "filtered": filtered})
        assert model.get_steps() == ["raw", "filtered"]
