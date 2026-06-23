import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from viewephys.data_model import SpikeGLXDataModel
from viewephys.gui import A_SCALAR, EphysBinViewer, create_app, viewephys
from viewephys.tests.test_viewer_helpers import synthetic_seismic_data
from viewephys.viewer.gui import EasyQC, viewseis


@pytest.fixture
def synthetic_seis():
    return synthetic_seismic_data()


@pytest.fixture
def easyqc_window(qtbot):
    window = EasyQC()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)
    yield window
    window.close()
    window.deleteLater()
    qtbot.wait(50)


@pytest.fixture
def view_with_data(qtbot, synthetic_seis):
    data, header = synthetic_seis
    window = viewseis(data, si=0.002, h=header, title="test")
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)
    yield window
    window.close()
    window.deleteLater()
    qtbot.wait(50)


def test_viewseis_shows(view_with_data):
    assert view_with_data.isVisible()
    assert hasattr(view_with_data, "plotItem_seismic")


def test_window_builds(easyqc_window):
    assert easyqc_window.isVisible()
    assert hasattr(easyqc_window, "plotItem_seismic")


def test_gain_edit_updates(view_with_data, qtbot):
    window = view_with_data
    window.lineEdit_gain.setText("6")
    qtbot.keyPress(window.lineEdit_gain, QtCore.Qt.Key_Return)
    qtbot.mouseClick(window.radio_wiggle, QtCore.Qt.LeftButton)
    qtbot.keyPress(window.lineEdit_gain, QtCore.Qt.Key_Return)
    assert float(window.lineEdit_gain.text()) == 6.0


def test_toggle_density_wiggle(view_with_data, qtbot):
    window = view_with_data
    assert window._display_mode == "density"
    qtbot.mouseClick(window.radio_wiggle, QtCore.Qt.LeftButton)
    assert window._display_mode == "wiggle"
    assert window.imageItem_seismic.image is None
    qtbot.mouseClick(window.radio_density, QtCore.Qt.LeftButton)
    assert window._display_mode == "density"


def test_sort_reorders_plotted_image_multiple_keys(view_with_data):
    """Sorting by multiple keys should reorder the plotted image
    without mutating the header.
    """
    window = view_with_data
    original_header = {
        key: values.copy() for key, values in window.model.header.items()
    }
    keys = ["receiver_number", "receiver_line"]
    expected_indices = np.lexsort(
        [window.model.header["receiver_number"], window.model.header["receiver_line"]]
    )
    expected_image = window.model.data[expected_indices, :]

    window.ctrl.sort(keys)

    np.testing.assert_array_equal(window.ctrl.trace_indices, expected_indices)
    np.testing.assert_array_equal(window.imageItem_seismic.image, expected_image)
    for key, original_values in original_header.items():
        np.testing.assert_array_equal(window.model.header[key], original_values)


def test_sort_reorders_plotted_image_descending(view_with_data):
    """Descending sort should reorder the plotted image."""
    window = view_with_data
    expected_indices = np.lexsort([-window.model.header["receiver_number"]])
    expected_image = window.model.data[expected_indices, :]

    window.ctrl.sort(["!receiver_number"])

    np.testing.assert_array_equal(window.ctrl.trace_indices, expected_indices)
    np.testing.assert_array_equal(window.imageItem_seismic.image, expected_image)


class _FakeSR:
    """Minimal stand-in for spikeglx.Reader, exposing only what the jump logic reads."""

    fs = 30000.0
    ns = 30000 * 2200  # 2200 s recording, like the screenshot in the request


class _FakeArraySR(_FakeSR):
    """Small array reader used to test jump reloads without a real binary file."""

    nc = 4
    nsync = 0
    type = "ap"
    geometry = {"trace": np.arange(nc)}

    def __getitem__(self, key):
        sample_slice, channel_slice = key
        first = 0 if sample_slice.start is None else sample_slice.start
        last = self.ns if sample_slice.stop is None else sample_slice.stop
        first_channel = 0 if channel_slice.start is None else channel_slice.start
        last_channel = self.nc if channel_slice.stop is None else channel_slice.stop
        shape = (last - first, last_channel - first_channel)
        return np.zeros(shape, dtype=np.float32)


class _RandomFakeArraySR(_FakeArraySR):
    """A small test file used to check the data displyaed is as expected."""

    ns = int(
        _FakeArraySR.fs * 5
    )  # because we create the full array, make short recording
    data = np.random.random((ns, _FakeArraySR.nc))

    def __getitem__(self, key):
        return self.data[key]


class _FakeViewBox:
    """Minimal view box that records range changes made by the jump code."""

    def __init__(self, view_range=None):
        self._view_range = view_range or ([0.0, 1.0], [0.0, 1.0])
        self.xrange = None
        self.yrange = None

    def viewRange(self):
        return self._view_range

    def setXRange(self, x0, x1, padding=0):
        self.xrange = (x0, x1, padding)

    def setYRange(self, y0, y1, padding=0):
        self.yrange = (y0, y1, padding)


class _FakeCtrl:
    """Minimal controller exposing data limits for range clamping."""

    def __init__(self, xlim):
        self._xlim = xlim

    def limits(self):
        return self._xlim, [0.0, 1.0]


class _FakeViewer:
    """Small viewer object used to test range preservation."""

    def __init__(self, xlim=None, view_range=None):
        self.viewBox_seismic = _FakeViewBox(view_range=view_range)
        self.ctrl = _FakeCtrl(xlim or [0.0, 1.0])

    def isVisible(self):
        return True

    def close(self):
        return None


def _centered_first_sample(typed_seconds, fs, ns, window_length_n):
    """Return the expected first sample after centering a jump request."""
    requested_sample = int(round(typed_seconds * fs))
    requested_sample = max(0, min(requested_sample, int(ns) - 1))
    max_first = max(0, int(ns) - window_length_n)
    first_sample = requested_sample - window_length_n // 2
    return max(0, min(first_sample, max_first))


@pytest.fixture
def jump_window(qtbot, monkeypatch):
    window = EphysBinViewer()
    qtbot.addWidget(window)
    window.data = SpikeGLXDataModel(_FakeSR())
    window._setup_viewers_and_checkboxes()
    slider_max = int(np.floor(window.data.get_num_samples() / window.window_length_n))
    window.horizontalSlider.setMaximum(slider_max)
    window.horizontalSlider.setEnabled(True)
    window.lineEdit_jumpTime.setEnabled(True)
    monkeypatch.setattr(
        window, "on_horizontalSliderReleased", lambda center_time=None: None
    )
    window.show()
    qtbot.wait(50)
    yield window
    window.close()
    window.deleteLater()
    qtbot.wait(50)


@pytest.mark.parametrize(
    "typed_seconds",
    [0.0, 0.4, 0.5, 100.0, 500.0, 500.150, 1173.67, 2199.99],
)
def test_jump_to_time_loads_exact_sample(jump_window, qtbot, typed_seconds):
    """Jump-to should center the window on the requested sample, while
    parking the slider near the loaded window for visual feedback only."""
    window = jump_window
    fs = window.data.get_sampling_frequency()
    expected_first = _centered_first_sample(
        typed_seconds, fs, window.data.get_num_samples(), window.window_length_n
    )
    expected_slider = max(
        0,
        min(
            int(round(expected_first / window.window_length_n)),
            window.horizontalSlider.maximum(),
        ),
    )
    expected_t = expected_first / fs

    window.lineEdit_jumpTime.setText(str(typed_seconds))
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)

    assert window._first_sample == expected_first
    assert window.horizontalSlider.value() == expected_slider
    assert window.lineEdit_jumpTime.text() == f"{expected_t:0.6f}"


def test_jump_to_time_non_chunk_aligned(jump_window, qtbot):
    """Sanity check that 500.150 s lands at the center of the loaded window."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    window.on_jumpToTimeRequested()

    requested_sample = int(round(500.150 * window.data.get_sampling_frequency()))
    assert window._first_sample + window.window_length_n // 2 == requested_sample
    assert window._first_sample == 14_999_500
    assert window.horizontalSlider.value() == 1500
    assert window.lineEdit_jumpTime.text() == "499.983333"


def test_slider_drag_resets_first_sample_to_chunk(jump_window, qtbot):
    """After a non-chunk-aligned jump, dragging the slider must snap
    `_first_sample` back to the chunk boundary so subsequent loads use the
    slider position."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window._first_sample == 14_999_500  # not chunk-aligned

    window.horizontalSlider.setValue(1501)
    assert window._first_sample == 1501 * window.window_length_n
    assert window.lineEdit_jumpTime.text() == (
        f"{1501 * window.window_length_n / window.data.get_sampling_frequency():0.6f}"
    )


def test_jump_to_time_clamps_out_of_range(jump_window, qtbot):
    window = jump_window
    max_first = max(0, int(window.data.get_num_samples()) - window.window_length_n)

    expected_slider_high = max(
        0,
        min(
            int(round(max_first / window.window_length_n)),
            window.horizontalSlider.maximum(),
        ),
    )

    window.lineEdit_jumpTime.setText("-50")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window._first_sample == 0
    assert window.horizontalSlider.value() == 0

    window.lineEdit_jumpTime.setText("99999")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window._first_sample == max_first
    assert window.horizontalSlider.value() == expected_slider_high


def test_jump_to_time_ignores_garbage(jump_window, qtbot):
    window = jump_window
    window.horizontalSlider.setValue(42)
    window.lineEdit_jumpTime.setText("abc")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window.horizontalSlider.value() == 42

    window.lineEdit_jumpTime.setText("")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window.horizontalSlider.value() == 42


def test_jump_to_time_recenters_existing_zoom(qtbot, monkeypatch):
    """Reloaded viewers should keep zoom width and recenter on the jump time."""
    window = EphysBinViewer()
    qtbot.addWidget(window)
    window.data = SpikeGLXDataModel(_FakeArraySR())
    window._setup_viewers_and_checkboxes()
    window.horizontalSlider.setMaximum(
        int(np.floor(window.data.get_num_samples() / window.window_length_n))
    )
    for checkbox in window.cbs.values():
        checkbox.setChecked(False)
    window.cbs["raw"].setChecked(True)
    window.viewers["raw"] = _FakeViewer(view_range=([400.0, 400.1], [10.0, 20.0]))

    captured = {}

    def fake_viewephys(data, fs, channels=None, title="ephys", t0=0.0, **kwargs):
        """Return a fake viewer with the same x-limits as the displayed chunk."""
        viewer = _FakeViewer(xlim=[t0, t0 + data.shape[1] / fs])
        captured["t0"] = t0
        captured["viewer"] = viewer
        return viewer

    monkeypatch.setattr("viewephys.gui.viewephys", fake_viewephys)

    window.lineEdit_jumpTime.setText("500.150")
    window.on_jumpToTimeRequested()

    x0, x1, padding = captured["viewer"].viewBox_seismic.xrange
    y0, y1, y_padding = captured["viewer"].viewBox_seismic.yrange
    assert captured["t0"] == pytest.approx(
        window._first_sample / window.data.get_sampling_frequency()
    )
    assert (x0 + x1) / 2 == pytest.approx(500.150)
    assert x1 - x0 == pytest.approx(0.1)
    assert padding == 0
    assert (y0, y1, y_padding) == (10.0, 20.0, 0)

    window.close()
    window.deleteLater()


def test_script_api_create_app(qtbot):
    """create_app() must be importable from viewephys.gui and return a QApplication.

    Regression guard for the README script-usage examples.
    """
    app = create_app()
    assert isinstance(app, QtWidgets.QApplication)


def test_script_api_viewephys(qtbot, synthetic_seis):
    """viewephys() must be importable from viewephys.gui and open a window.

    Regression guard for the README script-usage examples.
    """
    data, header = synthetic_seis
    window = viewephys(data, fs=30000, title="test_script_api")
    qtbot.addWidget(window)
    assert window is not None
    window.close()


def test_window_size_change_displays_different_data(qtbot):
    """Changing the window size must reload and display the matching chunk of
    the underlying recording in the spawned 'raw' viewer."""
    window = EphysBinViewer()
    qtbot.addWidget(window)
    window.data = SpikeGLXDataModel(_RandomFakeArraySR())
    window.update_slider_limits()
    for checkbox in window.cbs.values():
        checkbox.setChecked(False)
    window.cbs["raw"].setChecked(True)
    window.on_horizontalSliderReleased()

    # Display a 0.10 s window and check the raw viewer shows the matching chunk.
    window.lineEdit_windowSize.setText("0.10")
    window.on_lineEdit_windowSizeChanged()
    n0 = window.window_length_n
    expected0 = window.data.get_data(0, n0, "raw").T * A_SCALAR
    np.testing.assert_array_equal(
        window.viewers["raw"].imageItem_seismic.image, expected0
    )

    # A wider window must display a different, larger chunk of the recording.
    window.lineEdit_windowSize.setText("0.20")
    window.on_lineEdit_windowSizeChanged()
    n1 = window.window_length_n
    expected1 = window.data.get_data(0, n1, "raw").T * A_SCALAR
    np.testing.assert_array_equal(
        window.viewers["raw"].imageItem_seismic.image, expected1
    )
    assert window.viewers["raw"].imageItem_seismic.image.shape[0] == n1 != n0

    # Move the slider to start one second (fs samples) into the recording and
    # check the raw viewer now shows that later chunk.
    fs = window.data.get_sampling_frequency()
    slider_value = int(fs // n1)
    window.horizontalSlider.setValue(slider_value)
    window.on_horizontalSliderReleased()
    first = int(fs)
    assert window._first_sample == first
    expected_moved = window.data.get_data(first, first + n1, "raw").T * A_SCALAR
    np.testing.assert_array_equal(
        window.viewers["raw"].imageItem_seismic.image, expected_moved
    )

    window.close()
    window.deleteLater()


def test_auto_downsample_true_by_default(view_with_data):
    """Auto downsample is on by default, so the image item must downsample."""
    window = view_with_data
    assert window.actionAutoDownsample.isChecked() is True
    assert window._auto_downsample is True
    assert window.imageItem_seismic.autoDownsample is True


def test_auto_downsample_toggles_view_item(view_with_data):
    """Toggling the View menu item flips the image item's autoDownsample flag."""
    window = view_with_data
    assert window._display_mode == "density"

    window.actionAutoDownsample.setChecked(False)
    assert window._auto_downsample is False
    assert window.imageItem_seismic.autoDownsample is False

    window.actionAutoDownsample.setChecked(True)
    assert window._auto_downsample is True
    assert window.imageItem_seismic.autoDownsample is True
