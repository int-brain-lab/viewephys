from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np
import scipy.signal

from PyQt5 import QtWidgets, QtCore, QtGui, uic
import pyqtgraph as pg

from brainbox.processing import bincount2D
from brainbox.io.one import EphysSessionLoader
from brainbox.io.spikeglx import Streamer
import one.alf.io as alfio
from one.alf.files import get_session_path
import spikeglx
from ibldsp import voltage, utils
from iblatlas.atlas import BrainRegions

from viewephys.gui import viewephys, SNS_PALETTE

T_BIN = .007  # time bin size in secs
D_BIN = 10  # depth bin size in um
YMAX = 4000


def view_pid(pid, one):
    """
    Display an interactive raster plot for a given probe insertion
    :param pid:
    :param one:
    :return:
    """
    eid, pname = one.pid2eid(pid)
    sl = EphysSessionLoader(eid=eid, one=one)
    sl.load_trials()
    sl.load_spike_sorting(pnames=[pname])
    sr = Streamer(pid=pid, one=one)
    rv = RasterView()
    rv.set_model(sr, *sl.load_spike_sorting(dataset_types=['spikes.samples']), trials=sl.trials)
    return rv


def view_raster(bin_file):
    """
    Display an interactive raster plot for a given ap binary file on a local server
    Mostly deprecated as it requires a full session to be downloaded
    :param bin_file:
    :return:
    """
    bin_file = Path(bin_file)
    pname = bin_file.parent.name
    session_path = get_session_path(bin_file)
    ssl = EphysSessionLoader(session_path=session_path, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=['spikes.samples'])
    trials = alfio.load_object(ssl.session_path.joinpath('alf'), 'trials')
    return RasterView(bin_file, spikes, clusters, trials=trials)


@dataclass
class ProbeData:
    spikes: Union[dict, pd.DataFrame]
    clusters: Union[dict, pd.DataFrame]
    channels: Union[dict, pd.DataFrame] = field(default_factory=dict)
    trials: Union[dict, pd.DataFrame] = field(default_factory=dict)
    sr: Union[spikeglx.Reader, Streamer, str, Path] = None

    def __post_init__(self):
        if isinstance(self.sr, str) or isinstance(self.sr, Path):
            self.sr = spikeglx.Reader(self.ap_file)
        # TODO set the good units only ?
        # set the raster data
        iok = ~np.isnan(self.spikes.depths)
        self.raster, self.raster_times, self.raster_depths = bincount2D(
            self.spikes.times[iok], self.spikes.depths[iok], T_BIN, D_BIN)
        self.br = BrainRegions()


class RasterView(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        self.eqcs = []
        super(RasterView, self).__init__(*args, **kwargs)
        # wave by Diana Militano from the Noun Projectp
        uic.loadUi(Path(__file__).parent.joinpath('raster.ui'), self)
        background_color = self.palette().color(self.backgroundRole())
        self.plotItem_raster.setAspectLocked(False)
        self.imageItem_raster = pg.ImageItem()
        self.plotItem_raster.setBackground(background_color)
        self.plotItem_raster.addItem(self.imageItem_raster)
        self.viewBox_raster = self.plotItem_raster.getPlotItem().getViewBox()
        s = self.viewBox_raster.scene()
        # vb.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        s.sigMouseClicked.connect(self.mouseClick)
        self.show()
        self.actionopen.triggered.connect(self.open_file)
        self.settings = QtCore.QSettings('int-brain-lab', 'Raster')

    def set_model(self, sr, spikes, clusters, channels=None, trials=None):
        self.model = ProbeData(spikes, clusters, channels=channels, trials=trials, sr=sr)
        # set image
        self.imageItem_raster.setImage(self.data.raster.T)
        transform = [T_BIN, 0., 0., 0., D_BIN, 0., -.5, -.5, 1.]
        self.transform = np.array(transform).reshape((3, 3)).T
        self.imageItem_raster.setTransform(QtGui.QTransform(*transform))
        self.plotItem_raster.setLimits(xMin=0, xMax=self.data.raster_times[-1], yMin=0, yMax=self.data.raster_depths[-1])
        # set colormap
        cm = pg.colormap.get('Greys', source='matplotlib')  # prepare a linear color map
        bar = pg.ColorBarItem(values=(0, .5), colorMap=cm)  # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem(self.imageItem_raster)
        ################################################## plot location
        # self.view.layers[label] = {'layer': new_scatter, 'type': 'scatter'}
        self.line_eqc = pg.PlotCurveItem()
        self.plotItem_raster.addItem(self.line_eqc)
        # self.plotItem_raster.removeItem(new_curve)
        ################################################## plot trials
        if trials is not None:
            trial_times = dict(
                goCue_times=trials['goCue_times'],
                error_times=trials['feedback_times'][trials['feedbackType'] == -1],
                reward_times=trials['feedback_times'][trials['feedbackType'] == 1])
            self.trial_lines = {}
            for i, k in enumerate(trial_times):
                self.trial_lines[k] = pg.PlotCurveItem()
                self.plotItem_raster.addItem(self.trial_lines[k])
                x = np.tile(trial_times[k][:, np.newaxis], (1, 2)).flatten()
                y = np.tile(np.array([0, 1, 1, 0]), int(trial_times[k].shape[0] / 2 + 1))[:trial_times[k].shape[0] * 2] * YMAX
                couleur = np.r_[np.array(SNS_PALETTE[i]) * 255, 22].astype(np.uint8)
                print(couleur)
                self.trial_lines[k].setData(x=x.flatten(), y=y.flatten(), pen=pg.mkPen(couleur))
                # self.trial_lines[k].setVisible(False)

    def open_file(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption='Select Raw electrophysiology recording',
            directory=self.settings.value('bin_file_path'),
            filter='*.*bin')
        if file == '':
            return
        file = Path(file)
        self.settings.setValue("bin_file_path", str(file.parent))

    def mouseClick(self, event):
        """Draws a line on the raster and display in EasyQC"""
        if not event.double():
            return
        qxy = self.imageItem_raster.mapFromScene(event.scenePos())
        x = qxy.x()
        self.show_ephys(t0=self.data.raster_times[int(x - .5)])
        ymax = np.max(self.data.raster_depths) + 50
        self.line_eqc.setData(x=x + np.array([-.5, -.5, .5, .5]),
                              y=np.array([0, ymax, ymax, 0]),
                              pen=pg.mkPen((0, 255, 0)))

    def keyPressEvent(self, e):
        """
        page-up / ctrl + a :  gain up
        page-down / ctrl + z : gain down
        :param e:
        """
        k, m = (e.key(), e.modifiers())
        # page up / ctrl + a
        if k == QtCore.Qt.Key_PageUp or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_A):
            self.imageItem_raster.setLevels([0, self.imageItem_raster.levels[1] / 1.4])
        # page down / ctrl + z
        elif k == QtCore.Qt.Key_PageDown or (
                m == QtCore.Qt.ControlModifier and k == QtCore.Qt.Key_Z):
            self.imageItem_raster.setLevels([0, self.imageItem_raster.levels[1] * 1.4])

    def show_ephys(self, t0, tlen=.4):

        first = int(t0 * self.data.sr.fs)
        last = first + int(self.data.sr.fs * tlen)

        raw = self.data.sr[first:last, : - self.data.sr.nsync].T

        butter_kwargs = {'N': 3, 'Wn': 300 / self.data.sr.fs * 2, 'btype': 'highpass'}
        sos = scipy.signal.butter(**butter_kwargs, output='sos')
        butt = scipy.signal.sosfiltfilt(sos, raw)
        destripe = voltage.destripe(raw, fs=self.data.sr.fs, channel_labels=True)
        self.eqc_raw = viewephys(butt, self.data.sr.fs, channels=self.data.channels, br=self.data.br, title='butt', t0=t0, t_scalar=1)
        self.eqc_des = viewephys(destripe, self.data.sr.fs, channels=self.data.channels, br=self.data.br, title='destripe', t0=t0, t_scalar=1)
        stripes_noise = 20 * np.log10(np.median(utils.rms(butt - destripe)))
        eqc_xrange = [t0 + tlen / 2 - 0.01, t0 + tlen / 2 + 0.01]
        self.eqc_des.viewBox_seismic.setXRange(*eqc_xrange)
        self.eqc_raw.viewBox_seismic.setXRange(*eqc_xrange)

        # eqc2 = viewephys(butt - destripe, self.sr.fs, channels=None, br=None, title='diff')
        # overlay spikes
        tprobe = self.data.spikes.samples / self.data.sr.fs
        slice_spikes = slice(np.searchsorted(tprobe, t0), np.searchsorted(tprobe, t0 + tlen))
        t = tprobe[slice_spikes]
        c = self.data.clusters.channels[self.data.spikes.clusters[slice_spikes]]
        self.eqc_raw.ctrl.add_scatter(t, c)
        self.eqc_des.ctrl.add_scatter(t, c)
