from pathlib import Path

import numpy as np
import scipy.signal
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore, uic

from ibllib.io import spikeglx
from iblutil.numerical import ismember
from ibllib.ephys.neuropixel import trace_header
from ibllib.dsp import voltage
import easyqc.qt
from easyqc.gui import EasyQC

T_SCALAR = 1e3  # defaults ms for user side
A_SCALAR = 1e6  # defaults uV for user side
NSAMP_CHUNK = 10000  # window length in samples

SNS_PALETTE = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)]


class EphysBinViewer(QtWidgets.QMainWindow):
    def __init__(self, bin_file=None, *args, **kwargs):
        """
        :param parent:
        :param sr: ibllib.io.spikeglx.Reader instance
        """
        super(EphysBinViewer, self).__init__(*args, *kwargs)
        self.settings = QtCore.QSettings('int-brain-lab', 'EphysBinViewer')
        uic.loadUi(Path(__file__).parent.joinpath('nav_file.ui'), self)
        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent.joinpath('viewephys.svg'))))
        self.actionopen.triggered.connect(self.open_file)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setTickInterval(10)
        self.horizontalSlider.sliderReleased.connect(self.on_horizontalSliderReleased)
        self.horizontalSlider.valueChanged.connect(self.on_horizontalSliderValueChanged)
        self.label_smin.setText('0')
        self.show()
        self.viewers = {'butterworth': None, 'destripe': None}
        self.cbs = {'butterworth': self.cb_butterworth_ap, 'destripe': self.cb_destripe_ap}
        if bin_file is not None:
            self.open_file(file=bin_file)

    def open_file(self, *args, file=None):
        if file is None:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption='Select Raw electrophysiology recording',
                directory=self.settings.value('bin_file_path'), filter='*.*bin')
        if file == '':
            return
        file = Path(file)
        self.settings.setValue("path", str(file.parent))
        self.sr = spikeglx.Reader(file)
        # enable and set slider
        self.horizontalSlider.setMaximum(np.floor(self.sr.ns / NSAMP_CHUNK))
        tmax = np.floor(self.sr.ns / NSAMP_CHUNK) * NSAMP_CHUNK / self.sr.fs
        self.label_smax.setText(f"{tmax:0.2f}s")
        tlabel = f'{self.sr.file_bin.name} \n \n' \
                 f'NEUROPIXEL {self.sr.major_version} \n' \
                 f'{self.sr.rl} seconds long \n' \
                 f'{self.sr.fs} Hz Sampling Frequency \n' \
                 f'{self.sr.nc} Channels'
        self.label.setText(tlabel)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setEnabled(True)
        self.on_horizontalSliderReleased()

    def on_horizontalSliderValueChanged(self):
        tcur = self.horizontalSlider.value() * NSAMP_CHUNK / self.sr.fs
        self.label_sval.setText(f"{tcur:0.2f}s")

    def on_horizontalSliderReleased(self):
        first = int(float(self.horizontalSlider.value()) * NSAMP_CHUNK)
        last = first + int(NSAMP_CHUNK)
        data = self.sr[first:last, :-self.sr.nsync].T
        # get parameters for both AP and LFP band

        if self.sr.type == 'lf':
            butter_kwargs = {'N': 3, 'Wn': 3 / self.sr.fs * 2, 'btype': 'highpass'}
            fcn_destripe = voltage.destripe_lfp
        else:
            butter_kwargs = {'N': 3, 'Wn': 300 / self.sr.fs * 2, 'btype': 'highpass'}
            fcn_destripe = voltage.destripe
        sos = scipy.signal.butter(**butter_kwargs, output='sos')
        data = scipy.signal.sosfiltfilt(sos, data)
        t0 = first / self.sr.fs * 0
        for k in self.viewers:
            if not self.cbs[k].isChecked():
                continue
            if k == 'destripe':
                data = fcn_destripe(x=data, fs=self.sr.fs, channel_labels=True, neuropixel_version=self.sr.major_version)
            self.viewers[k] = viewephys(data, self.sr.fs, title=k, t0=t0 * T_SCALAR, t_scalar=T_SCALAR, a_scalar=A_SCALAR)

    def closeEvent(self, event):
        for k in self.viewers:
            ev = self.viewers[k]
            if ev is not None:
                ev.close()
        self.close()


class EphysViewer(EasyQC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl.model.picks = {'sample': np.array([]), 'trace': np.array([]), 'amp': np.array([])}
        self.menufile.setEnabled(True)
        self.settings = QtCore.QSettings('int-brain-lab', 'EphysViewer')
        self.header_curves = {}
        self.menupick = self.menuBar().addMenu('&Pick')
        self.action_pick = QtWidgets.QAction('Pick', self)
        self.action_pick.setCheckable(True)
        self.menupick.addAction(self.action_pick)
        self.action_pick.triggered.connect(self.menu_pick_callback)
        self.show()

    @staticmethod
    def _get_or_create(title=None):
        ev = next(filter(lambda e: e.isVisible() and e.windowTitle() == title,
                         EphysViewer._instances()), None)
        if ev is None:
            ev = EphysViewer()
            ev.setWindowTitle(title)
        return ev

    def rm_header_curve(self, name):
        if name not in self.header_curves:
            return
        curve = self.header_curves.pop(name)
        self.plotItem_header_h.removeItem(curve)

    def add_header_times(self, times, name):
        """
        Adds behaviour events in the horizontal header axes. Wra[s the add_header_curve method
        :param times: np.array , vector of times
        :param name: string
        :return:
        """
        y = np.tile(np.array([0, 1, np.nan]), times.size)
        x = np.tile(times[:, np.newaxis] * T_SCALAR, 3).flatten()
        self.add_header_curve(x, y, name)

    def add_header_curve(self, x, y, name):
        """
        Adds a plot in the horizontal header axes linked to the image display. The x-axis
        represents times and is linked to the image display
        :param x:
        :param y:
        :param name:
        :return:
        """
        if name in self.header_curves:
            self.rm_header_curve(name)
        ind = len(self.header_curves.keys())
        pen = pg.mkPen(color=np.array(SNS_PALETTE[ind]) * 255)
        self.header_curves[name] = pg.PlotCurveItem(x=x, y=y, connect='finite', pen=pen, name='licks')
        self.plotItem_header_h.addItem(self.header_curves[name])

    def menu_pick_callback(self, event):
        # disable the picking
        if self.action_pick.isChecked():
            self.viewBox_seismic.scene().sigMouseClicked.connect(self.mouseClickPickingEvent)
        else:
            self.viewBox_seismic.scene().sigMouseClicked.disconnect(self.mouseClickPickingEvent)

    def mouseClickPickingEvent(self, event):
        """
        When the pick action is enabled this is triggered on mouse click
        - left button click sets a point
        - middle button removes a point
        - control + left does not wrap on maximum around pick
        """
        TR_RANGE = 3
        S_RANGE = int(0.5 / self.ctrl.model.si)
        qxy = self.imageItem_seismic.mapFromScene(event.scenePos())
        s, tr = (qxy.x(), qxy.y())
        if event.buttons() == QtCore.Qt.MiddleButton:
            iclose = np.where(np.logical_and(
                np.abs(self.ctrl.model.picks['sample'] - s) <= (S_RANGE + 1),
                np.abs(self.ctrl.model.picks['trace'] - tr) <= (TR_RANGE + 1)
            ))[0]
            self.ctrl.model.picks = {k: np.delete(self.ctrl.model.picks[k], iclose) for k in self.ctrl.model.picks}

        elif event.buttons() == QtCore.Qt.LeftButton:
            if event.modifiers() == QtCore.Qt.ControlModifier:
                tmax, xmax = (int(round(s)), int(round(tr)))
            else:
                xscale = np.arange(-TR_RANGE, TR_RANGE + 1) + np.round(tr).astype(np.int32)
                tscale = np.arange(-S_RANGE, S_RANGE + 1) + np.round(s).astype(np.int32)
                ix = slice(xscale[0], xscale[-1] + 1)
                it = slice(tscale[0], tscale[-1] + 1)

                out_of_tr_range = xscale[0] < 0 or xscale[-1] > (self.ctrl.model.ntr - 1)
                out_of_time_range = tscale[0] < 0 or tscale[-1] > (self.ctrl.model.ns - 1)

                if out_of_time_range or out_of_tr_range:
                    print(xscale, tscale)
                    return
                tmax, xmax = np.unravel_index(np.argmax(np.abs(self.ctrl.model.data[it, ix])),
                                              (S_RANGE * 2 + 1, TR_RANGE * 2 + 1))
                tmax, xmax = (tscale[tmax], xscale[xmax])

            self.ctrl.model.picks['sample'] = np.r_[self.ctrl.model.picks['sample'], tmax]
            self.ctrl.model.picks['trace'] = np.r_[self.ctrl.model.picks['trace'], xmax]
            self.ctrl.model.picks['amp'] = np.r_[self.ctrl.model.picks['amp'], self.ctrl.model.data[tmax, xmax]]
        # updates scatter plot
        self.ctrl.add_scatter(self.ctrl.model.picks['sample'] * self.ctrl.model.si,
                              self.ctrl.model.picks['trace'],
                              label='_picks', rgb=(0, 255, 255))


def viewephys(data, fs, channels=None, br=None, title='ephys', t0=0, t_scalar=T_SCALAR, a_scalar=A_SCALAR):
    """
    :param data: [nc, ns]
    :param fs:
    :param channels:
    :param br:
    :param title:
    :return:
    """

    easyqc.qt.create_app()
    ev = EphysViewer._get_or_create(title=title)

    if channels is None or br is None:
        channels = trace_header(version=1)
    if data is not None:
        ev.ctrl.update_data(data.T * a_scalar, si=1 / fs * t_scalar, h=channels, taxis=0, t0=t0)
    if br is not None and 'atlas_id' in channels:
        _, ir = ismember(channels['atlas_id'], br.id)
        image = br.rgb[ir].astype(np.uint8)
        image = image[np.newaxis, :, :]
        imitem = pg.ImageItem(image)
        ev.plotItem_header_v.addItem(imitem)
        transform = [1, 0, 0, 0, 1, 0, -0.5, 0, 1.]
        imitem.setTransform(QtGui.QTransform(*transform))
        ev.plotItem_header_v.setLimits(xMin=-.5, xMax=.5)

    ev.show()
    return ev
