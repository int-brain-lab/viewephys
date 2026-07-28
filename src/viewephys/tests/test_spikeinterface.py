import numpy as np
import pytest
import spikeinterface.core as si_core
import spikeinterface.preprocessing as si_prepro

from viewephys.gui import A_SCALAR, EphysBinViewer

# The SpikeInterface recordings below are shifted onto a non-zero clock so the
# tests exercise the sample<->time mapping (t0/labels), not just frame slicing.
T_START = 40.0


def test_spikeinterface_checkboxes_show_selected_recordings(qtbot):
    # Create a small in-memory SpikeInterface recording for the viewer input.
    raw = si_core.generate_recording(
        num_channels=4,
        sampling_frequency=30_000.0,
        durations=[0.2],
        set_probe=False,
        seed=0,
    )

    # Shift onto a non-zero clock (mimics an SI recording that does not start at 0).
    raw.set_times(raw.get_times() + T_START)

    # Add preprocessing steps, preserving dict order as the expected UI order.
    bandpass = si_prepro.bandpass_filter(raw, freq_min=125.0, freq_max=3000)
    cmr = si_prepro.common_reference(bandpass, operator="median")
    recordings = {"raw": raw, "bandpass": bandpass, "cmr": cmr}

    # Load the dict and run the same setup helper used by the public constructor.
    window = EphysBinViewer(recordings)
    window.window_length_n = 64
    window.update_slider_limits()
    fs = window.data.get_sampling_frequency()

    # Check that dynamic checkboxes mirror the recording dict order and labels.
    assert list(window.cbs) == list(recordings)
    assert [checkbox.text() for checkbox in window.cbs.values()] == list(recordings)
    assert [checkbox.isChecked() for checkbox in window.cbs.values()] == [
        True,
        False,
        False,
    ]

    # Select every recording and move the viewed window away from the first chunk.
    for checkbox in window.cbs.values():
        checkbox.setChecked(True)

    window.horizontalSlider.setValue(1)
    window.on_horizontalSliderReleased()

    first = window.window_length_n
    last = first + window.window_length_n
    assert list(window.viewers) == list(recordings)
    assert all(viewer is not None for viewer in window.viewers.values())

    # The window content is selected by sample index, so the frame slice is
    # unaffected by the non-zero clock: each viewer shows the matching traces.
    for step, viewer in window.viewers.items():
        expected = (
            recordings[step].get_traces(
                start_frame=first,
                end_frame=last,
                return_in_uV=True,
                segment_index=0,
            )
            * A_SCALAR
        )
        np.testing.assert_allclose(
            viewer.imageItem_seismic.image, expected, rtol=0, atol=1e-8
        )

    # The non-zero SI clock must be honored for display: the sample<->time helper
    # and every spawned viewer's time origin reflect the +T_START offset.
    expected_t0 = first / fs + T_START
    assert window.data.get_time_from_sample(first) == pytest.approx(expected_t0)
    assert window.lineEdit_jumpTime.text() == f"{expected_t0:0.6f}"
    for viewer in window.viewers.values():
        assert viewer.model.t0 == pytest.approx(expected_t0)

    window.close()


def test_spikeinterface_jump_to_time_uses_si_clock(qtbot):
    """Jumping to a typed time must convert through the SI clock.

    With a non-zero start time, ``get_sample_from_time`` and the old zero-origin
    ``t * fs`` conversion land on very different samples, so this guards against
    the jump handler regressing to plain ``t * fs``.
    """
    fs = 30_000.0
    raw = si_core.generate_recording(
        num_channels=4,
        sampling_frequency=fs,
        durations=[0.2],
        set_probe=False,
        seed=0,
    )
    raw.set_times(raw.get_times() + T_START)

    window = EphysBinViewer({"raw": raw})
    window.window_length_n = 64
    window.update_slider_limits()
    # Do not spawn viewer windows; we only assert the navigation maths.
    for checkbox in window.cbs.values():
        checkbox.setChecked(False)

    ns = raw.get_num_samples()
    max_first = max(0, ns - window.window_length_n)

    # Jump 0.05 s into the recording on the SI clock (i.e. T_START + 0.05 s).
    target_time = T_START + 0.05
    window.lineEdit_jumpTime.setText(f"{target_time:.6f}")
    window.on_jumpToTimeRequested()

    # Expected window start from the correct SI-clock conversion...
    requested = min(max(0, window.data.get_sample_from_time(target_time)), ns - 1)
    expected_first = min(max(0, requested - window.window_length_n // 2), max_first)

    assert window._first_sample == expected_first

    window.close()
