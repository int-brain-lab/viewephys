import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

from viewephys.gui import NSAMP_CHUNK, EphysBinViewer, create_app, viewephys
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


def _centered_first_sample(typed_seconds, fs, ns):
    """Return the expected first sample after centering a jump request."""
    requested_sample = int(round(typed_seconds * fs))
    requested_sample = max(0, min(requested_sample, int(ns) - 1))
    max_first = max(0, int(ns) - NSAMP_CHUNK)
    first_sample = requested_sample - NSAMP_CHUNK // 2
    return max(0, min(first_sample, max_first))


@pytest.fixture
def jump_window(qtbot, monkeypatch):
    window = EphysBinViewer()
    qtbot.addWidget(window)
    window.sr = _FakeSR()
    slider_max = int(np.floor(window.sr.ns / NSAMP_CHUNK))
    window.horizontalSlider.setMaximum(slider_max)
    window.horizontalSlider.setEnabled(True)
    window.lineEdit_jumpTime.setEnabled(True)
    window.pushButton_jumpTime.setEnabled(True)
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
    fs = window.sr.fs
    expected_first = _centered_first_sample(typed_seconds, fs, window.sr.ns)
    expected_slider = max(
        0,
        min(
            int(round(expected_first / NSAMP_CHUNK)), window.horizontalSlider.maximum()
        ),
    )
    expected_t = expected_first / fs

    window.lineEdit_jumpTime.setText(str(typed_seconds))
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)

    assert window._first_sample == expected_first
    assert window.horizontalSlider.value() == expected_slider
    assert window.label_sval.text() == f"{expected_t:0.6f}s"


def test_jump_to_time_non_chunk_aligned(jump_window, qtbot):
    """Sanity check that 500.150 s lands at the center of the loaded window."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    window.on_jumpToTimeRequested()

    requested_sample = int(round(500.150 * window.sr.fs))
    assert window._first_sample + NSAMP_CHUNK // 2 == requested_sample
    assert window._first_sample == 14_999_500
    assert window.horizontalSlider.value() == 1500
    assert window.label_sval.text() == "499.983333s"


def test_slider_drag_resets_first_sample_to_chunk(jump_window, qtbot):
    """After a non-chunk-aligned jump, dragging the slider must snap
    `_first_sample` back to the chunk boundary so subsequent loads use the
    slider position."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window._first_sample == 14_999_500  # not chunk-aligned

    window.horizontalSlider.setValue(1501)
    assert window._first_sample == 1501 * NSAMP_CHUNK
    assert window.label_sval.text() == f"{1501 * NSAMP_CHUNK / window.sr.fs:0.6f}s"


def test_jump_to_time_clamps_out_of_range(jump_window, qtbot):
    window = jump_window
    max_first = max(0, int(window.sr.ns) - NSAMP_CHUNK)
    expected_slider_high = max(
        0, min(int(round(max_first / NSAMP_CHUNK)), window.horizontalSlider.maximum())
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
    window.sr = _FakeArraySR()
    window.horizontalSlider.setMaximum(int(np.floor(window.sr.ns / NSAMP_CHUNK)))
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
    assert captured["t0"] == pytest.approx(window._first_sample / window.sr.fs)
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
