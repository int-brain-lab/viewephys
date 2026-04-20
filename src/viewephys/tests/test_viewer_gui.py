import pytest
from qtpy import QtCore

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
