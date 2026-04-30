import numpy as np
import pytest
from qtpy import QtCore

from viewephys.gui import NSAMP_CHUNK, EphysBinViewer
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
    monkeypatch.setattr(window, "on_horizontalSliderReleased", lambda: None)
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
    """Jump-to should load the window starting at the closest sample, while
    parking the slider at the nearest chunk for visual feedback only."""
    window = jump_window
    fs = window.sr.fs
    max_first = max(0, int(window.sr.ns) - NSAMP_CHUNK)
    expected_first = int(round(typed_seconds * fs))
    expected_first = max(0, min(expected_first, max_first))
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
    """Sanity check that 500.150 s lands on its exact sample, not the
    nearest 10000-sample chunk (which would be 500.000 or 500.333)."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)

    assert window._first_sample == int(round(500.150 * window.sr.fs))
    assert window._first_sample == 15_004_500
    assert window.horizontalSlider.value() == 1500
    assert window.label_sval.text() == "500.150000s"


def test_slider_drag_resets_first_sample_to_chunk(jump_window, qtbot):
    """After a non-chunk-aligned jump, dragging the slider must snap
    `_first_sample` back to the chunk boundary so subsequent loads use the
    slider position."""
    window = jump_window
    window.lineEdit_jumpTime.setText("500.150")
    qtbot.keyPress(window.lineEdit_jumpTime, QtCore.Qt.Key_Return)
    assert window._first_sample == 15_004_500  # not chunk-aligned

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
