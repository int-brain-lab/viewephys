import numpy as np
import spikeinterface.core as si_core
import spikeinterface.preprocessing as si_prepro

from viewephys.gui import A_SCALAR, EphysBinViewer


def test_spikeinterface_checkboxes_show_selected_recordings(qtbot):
    # Create a small in-memory SpikeInterface recording for the viewer input.
    raw = si_core.generate_recording(
        num_channels=4,
        sampling_frequency=30_000.0,
        durations=[0.2],
        set_probe=False,
        seed=0,
    )

    # Add preprocessing steps, preserving dict order as the expected UI order.
    bandpass = si_prepro.bandpass_filter(raw, freq_min=125.0, freq_max=3000)
    cmr = si_prepro.common_reference(bandpass, operator="median")
    recordings = {"raw": raw, "bandpass": bandpass, "cmr": cmr}

    # Load the dict and run the same setup helper used by the public constructor.
    window = EphysBinViewer(recordings)
    window.window_length_n = 64
    window.update_slider_limits()

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

    # Verify each spawned viewer displays the matching SpikeInterface trace slice.
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

    window.close()
