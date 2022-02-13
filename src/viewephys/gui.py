from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore


from iblutil.numerical import ismember
from ibllib.ephys.neuropixel import trace_header
import easyqc.qt
from easyqc.gui import EasyQC


class EphysViewer(EasyQC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.menufile.setEnabled(True)
        self.actionopen.triggered.connect(self.open_file)
        self.settings = QtCore.QSettings('int-brain-lab', 'EphysViewer')

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self, caption='Select Raw electrophysiology recording', directory=self.settings.value('bin_file_path'), filter='*.*bin')
        if file == '':
            return
        file = Path(file)
        self.settings.setValue("path", str(file.parent))

    @staticmethod
    def _get_or_create(title=None):
        ev = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         EphysViewer._instances()), None)
        if ev is None:
            ev = EphysViewer()
            ev.setWindowTitle(title)
        return ev


def viewephys(data, fs, channels=None, br=None, title='ephys', t0=0, t_scalar=1e3, a_scalar=1e6):
    """
    :param data: [nc, ns]
    :param fs:
    :param channels:
    :param br:
    :param title:
    :return:
    """

    app = easyqc.qt.create_app()
    ev = EphysViewer._get_or_create(title=title)

    if channels is None or br is None:
        channels = trace_header(version = 1)
    if data is not None:
        ev.ctrl.update_data(data.T * a_scalar, si=1 / fs * t_scalar, h=channels, taxis=0, t0=t0)
    if br is not None:
        _, ir = ismember(channels['atlas_id'], br.id)
        image = br.rgb[ir].astype(np.uint8)
        image = image[np.newaxis, :, :]
        imitem = pg.ImageItem(image)
        ev.plotItem_header_v.addItem(imitem)
        transform = [1, 0, 0, 0, 1, 0, -0.5, 0, 1.]
        imitem.setTransform(QtGui.QTransform(*transform))
        ev.plotItem_header_v.setLimits(xMin=-.5, xMax=.5)

    ev.QT_APP = app
    ev.show()
    return ev
