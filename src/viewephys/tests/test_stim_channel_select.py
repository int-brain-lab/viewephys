import numpy as np
import pytest

from viewephys.stim_artefact import StimArtefactViewer
from viewephys.tests.test_viewer_helpers import synthetic_seismic_data


def test_filter_indices_none_returns_full():
    full = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    out = StimArtefactViewer._filter_indices(full, None)
    np.testing.assert_array_equal(out, full)


def test_filter_indices_keeps_sorted_order():
    full = np.array([3, 1, 4, 0, 5], dtype=np.int64)
    out = StimArtefactViewer._filter_indices(full, {0, 4})
    # Order must come from ``full``, not the selection set.
    np.testing.assert_array_equal(out, np.array([4, 0]))


def test_filter_indices_empty_intersection_falls_back_to_full():
    full = np.array([3, 1, 4], dtype=np.int64)
    out = StimArtefactViewer._filter_indices(full, {99})
    np.testing.assert_array_equal(out, full)


@pytest.fixture
def stim_viewer(qtbot):
    data, header = synthetic_seismic_data(ntr=8, ns=800)
    window = StimArtefactViewer()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(20)
    window.model.set_data(data, si=0.001, header=header, t0=0.0, taxis=1)
    window.ctrl.set_model()
    window.refresh_channel_state()
    yield window
    window.close()
    window.deleteLater()
    qtbot.wait(20)


def test_channel_list_populated_from_sorted_indices(stim_viewer):
    n = stim_viewer.listWidget_stim_channels.count()
    assert n == stim_viewer._full_sorted_indices.size == 8
    # Default: every channel selected and "None" filter is active.
    assert stim_viewer._selected_traces is None
    assert stim_viewer.label_stim_channel_count.text() == "8 / 8 channels"


def test_apply_channel_subset_filters_displayed_traces(stim_viewer):
    lw = stim_viewer.listWidget_stim_channels
    lw.clearSelection()
    # Pick the first three list items (in sorted order).
    chosen_originals = []
    from qtpy import QtCore

    for i in range(3):
        item = lw.item(i)
        item.setSelected(True)
        chosen_originals.append(int(item.data(QtCore.Qt.UserRole)))

    stim_viewer._on_channels_apply()

    assert stim_viewer._selected_traces == set(chosen_originals)
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices,
        stim_viewer._full_sorted_indices[:3],
    )
    assert stim_viewer.label_stim_channel_count.text() == "3 / 8 channels"


def test_apply_all_resets_selection_to_none(stim_viewer):
    stim_viewer._selected_traces = {0, 1}
    stim_viewer._on_channels_all()
    stim_viewer._on_channels_apply()
    assert stim_viewer._selected_traces is None
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices, stim_viewer._full_sorted_indices
    )


def test_empty_selection_falls_back_to_full(stim_viewer):
    stim_viewer._on_channels_none()
    stim_viewer._on_channels_apply()
    # No items selected -> treated as "all" by Apply, so selected_traces is None.
    assert stim_viewer._selected_traces is None
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices, stim_viewer._full_sorted_indices
    )


def test_apply_in_wiggle_mode_updates_both_controllers(stim_viewer):
    from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE
    from qtpy import QtCore

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    # set_display_mode resets trace_indices via set_model -> recapture state.
    stim_viewer.refresh_channel_state()

    lw = stim_viewer.listWidget_stim_channels
    lw.clearSelection()
    chosen_originals = []
    for i in range(2):
        item = lw.item(i)
        item.setSelected(True)
        chosen_originals.append(int(item.data(QtCore.Qt.UserRole)))

    stim_viewer._on_channels_apply()

    expected = stim_viewer._full_sorted_indices[:2]
    # Both controllers must carry the filter so toggling display mode preserves it.
    np.testing.assert_array_equal(stim_viewer._ctrl_wiggle.trace_indices, expected)
    np.testing.assert_array_equal(stim_viewer._ctrl_image.trace_indices, expected)
    assert stim_viewer._selected_traces == set(chosen_originals)


def test_apply_in_wiggle_autospace_does_not_clamp_yrange(stim_viewer):
    from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE
    from qtpy import QtCore

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    stim_viewer.checkBox_wiggle_autospace.setChecked(True)
    stim_viewer.refresh_channel_state()

    lw = stim_viewer.listWidget_stim_channels
    lw.clearSelection()
    for i in range(3):
        lw.item(i).setSelected(True)

    stim_viewer._on_channels_apply()

    # In autospace mode, ``ControllerWiggle`` sets a y-range based on the
    # trace amplitude, not on integer slot indices. Verify our channel
    # filter did NOT clamp the view to ``[-0.5, n - 0.5]`` (which would
    # collapse the wiggle plot).
    ymin, ymax = stim_viewer.viewBox_seismic.viewRange()[1]
    n = 3
    assert not (ymin == -0.5 and ymax == n - 0.5), (
        "Wiggle autospace y-range was clamped to image-mode integer slots"
    )
    # Header trace_indices size must match the filter.
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices, stim_viewer._full_sorted_indices[:3]
    )


def test_switching_display_mode_preserves_selection(stim_viewer):
    from viewephys.viewer.gui import DISPLAY_MODE_DENSITY, DISPLAY_MODE_WIGGLE
    from qtpy import QtCore

    # Apply a subset in density (default) mode.
    lw = stim_viewer.listWidget_stim_channels
    lw.clearSelection()
    chosen_originals = []
    for i in range(3):
        item = lw.item(i)
        item.setSelected(True)
        chosen_originals.append(int(item.data(QtCore.Qt.UserRole)))
    stim_viewer._on_channels_apply()
    assert stim_viewer._selected_traces == set(chosen_originals)

    # Density -> wiggle: selection survives.
    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    assert stim_viewer._selected_traces == set(chosen_originals)
    assert stim_viewer.label_stim_channel_count.text() == "3 / 8 channels"
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices,
        stim_viewer._full_sorted_indices[
            np.isin(
                stim_viewer._full_sorted_indices,
                np.fromiter(chosen_originals, dtype=np.int64),
            )
        ],
    )

    # Wiggle -> density: still survives.
    stim_viewer.set_display_mode(DISPLAY_MODE_DENSITY)
    assert stim_viewer._selected_traces == set(chosen_originals)
    np.testing.assert_array_equal(
        stim_viewer.ctrl.trace_indices,
        stim_viewer._full_sorted_indices[
            np.isin(
                stim_viewer._full_sorted_indices,
                np.fromiter(chosen_originals, dtype=np.int64),
            )
        ],
    )


def test_toggling_autospace_preserves_channel_selection(stim_viewer):
    from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    stim_viewer.refresh_channel_state()

    lw = stim_viewer.listWidget_stim_channels
    lw.clearSelection()
    for i in range(3):
        lw.item(i).setSelected(True)
    stim_viewer._on_channels_apply()
    expected = stim_viewer._full_sorted_indices[:3]
    np.testing.assert_array_equal(stim_viewer.ctrl.trace_indices, expected)

    # Toggle autospace ON: filter must survive, not snap back to all 8.
    stim_viewer.checkBox_wiggle_autospace.setChecked(True)
    np.testing.assert_array_equal(stim_viewer.ctrl.trace_indices, expected)
    assert stim_viewer._selected_traces is not None

    # Toggle autospace OFF: same story.
    stim_viewer.checkBox_wiggle_autospace.setChecked(False)
    np.testing.assert_array_equal(stim_viewer.ctrl.trace_indices, expected)


def test_wiggle_view_locked_to_trace_bounds(stim_viewer):
    from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    stim_viewer.refresh_channel_state()

    n = int(stim_viewer._full_sorted_indices.size)
    # Read the y-limits the seismic plot enforces. They must bracket the
    # trace baselines (``[-0.5, n - 0.5]``) with a small symmetric padding
    # rather than the unbounded ``±inf`` we use to release image-mode clamps.
    state = stim_viewer.plotItem_seismic.getViewBox().state["limits"]
    ymin, ymax = state["yLimits"]
    assert np.isfinite(ymin) and np.isfinite(ymax)
    assert ymin <= -0.5 and ymax >= n - 0.5
    span = ymax - ymin
    full_span = (n - 0.5) - (-0.5)
    # Padding must stay reasonable relative to the trace span (≤ ~50%).
    assert span <= full_span * 1.5


def test_switching_to_wiggle_fully_zooms_out(stim_viewer):
    """After density->wiggle, the view must show the full data extent.

    Reproduces the bug where switching modes left the y-range at whatever
    sub-window the user had panned to in density mode (e.g. ``[2, 5]``
    instead of the full padded bounds). The view range must match the
    enforced y-limits exactly so the user is fully zoomed out.
    """
    from viewephys.viewer.gui import DISPLAY_MODE_DENSITY, DISPLAY_MODE_WIGGLE

    # Pre-zoom in density mode so the view is NOT showing all 8 traces.
    stim_viewer.set_display_mode(DISPLAY_MODE_DENSITY)
    stim_viewer.viewBox_seismic.setYRange(2.0, 5.0, padding=0)
    stim_viewer.viewBox_seismic.setXRange(0.05, 0.15, padding=0)

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)

    # Y view-range must equal the y-limits (fully zoomed out).
    ymin, ymax = stim_viewer.viewBox_seismic.viewRange()[1]
    state = stim_viewer.plotItem_seismic.getViewBox().state["limits"]
    lymin, lymax = state["yLimits"]
    assert ymin == pytest.approx(lymin, abs=1e-6)
    assert ymax == pytest.approx(lymax, abs=1e-6)

    # X must span the full data window.
    xmin, xmax = stim_viewer.viewBox_seismic.viewRange()[0]
    ns, si, t0 = stim_viewer.model.ns, stim_viewer.model.si, stim_viewer.model.t0
    assert xmin == pytest.approx(t0, abs=1e-6)
    assert xmax == pytest.approx(t0 + ns * si, abs=1e-6)


def test_toggling_autospace_in_wiggle_fully_zooms_out(stim_viewer):
    """Flipping auto-space ON/OFF must reset to the full padded extent."""
    from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE

    stim_viewer.set_display_mode(DISPLAY_MODE_WIGGLE)
    # Pan/zoom into a sliver.
    stim_viewer.viewBox_seismic.setYRange(1.0, 2.0, padding=0)

    stim_viewer.checkBox_wiggle_autospace.setChecked(True)
    ymin, ymax = stim_viewer.viewBox_seismic.viewRange()[1]
    state = stim_viewer.plotItem_seismic.getViewBox().state["limits"]
    lymin, lymax = state["yLimits"]
    assert ymin == pytest.approx(lymin, abs=1e-6)
    assert ymax == pytest.approx(lymax, abs=1e-6)

    # Toggle back off -> integer slot range, fully zoomed out.
    stim_viewer.checkBox_wiggle_autospace.setChecked(False)
    n = int(stim_viewer._full_sorted_indices.size)
    ymin, ymax = stim_viewer.viewBox_seismic.viewRange()[1]
    state = stim_viewer.plotItem_seismic.getViewBox().state["limits"]
    lymin, lymax = state["yLimits"]
    assert ymin == pytest.approx(lymin, abs=1e-6)
    assert ymax == pytest.approx(lymax, abs=1e-6)
    # Non-autospace pad is half a trace, so limits must extend at least
    # half a slot beyond [-0.5, n-0.5].
    assert lymin <= -0.5 - 0.5 + 1e-6
    assert lymax >= n - 0.5 + 0.5 - 1e-6
