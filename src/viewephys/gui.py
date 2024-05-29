from pathlib import Path

import pandas as pd
import numpy as np
import scipy.signal
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore, uic

import spikeglx
from neuropixel import trace_header
from ibldsp import voltage
from iblutil.numerical import ismember
import easyqc.qt
from easyqc.gui import EasyQC

T_SCALAR = 1e3  # defaults ms for user side
A_SCALAR = 1e6  # defaults uV for user side
NSAMP_CHUNK = 10000  # window length in samples
N_SAMPLES_INIT = 2000  # number of samples in the manual pick array

PICK_COLOR = (0, 255, 255)

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
        self.actionopen_live_recording.triggered.connect(self.open_file_live)
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

    def open_file_live(self, *args, **kwargs):
        self.open_file(*args, live=True, **kwargs)

    def open_file(self, *args, live=False, file=None):
        if file is None:
            file, _ = QtWidgets.QFileDialog.getOpenFileName(
                parent=self, caption='Select Raw electrophysiology recording',
                directory=self.settings.value('bin_file_path'), filter='*.*bin')
        if file == '':
            return
        file = Path(file)
        self.settings.setValue("bin_file_path", str(file.parent))
        ReaderClass = spikeglx.Reader if not live else spikeglx.OnlineReader
        try:
            self.sr = ReaderClass(file)
        except AssertionError:
            self.sr = spikeglx.Reader(file, dtype='int16', nc=384, fs=30000, ns=file.stat().st_size / 384 / 2)
        # enable and set slider
        self.horizontalSlider.setMaximum(int(np.floor(self.sr.ns / NSAMP_CHUNK)))
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
                data = fcn_destripe(x=data, fs=self.sr.fs, channel_labels=True, h=self.sr.geometry, neuropixel_version=self.sr.major_version)
            self.viewers[k] = viewephys(data, self.sr.fs, channels=self.sr.geometry, title=k, t0=t0 * T_SCALAR, t_scalar=T_SCALAR, a_scalar=A_SCALAR)

    def closeEvent(self, event):
        for k in self.viewers:
            ev = self.viewers[k]
            if ev is not None:
                ev.close()
        self.close()


class PickSpikes():

    def __init__(self):
        default_df = self.init_df()
        self.update_pick(default_df)

    def init_df(self, nrow=0):
        init_df = pd.DataFrame({
            'sample': np.zeros(nrow, dtype=np.int32),
            'trace': np.zeros(nrow, dtype=np.int32) * -1,
            'amp': np.zeros(nrow, dtype=np.int32),
            'group': np.zeros(nrow, dtype=np.int32),
        })
        return init_df

    def update_pick(self, df):
        self.picks = df
        self.pick_index = df.shape[0]  # Last index of spike picked (== len of df table)
        self.pick_group = df['group'].max()  # Last group created

    def load_df(self, df):
        '''
        Load a dataframe that contains already picked spikes
        :return:
        '''
        default_df = self.init_df()

        if isinstance(df, pd.DataFrame):
            # check all keys are in
            indxmissing = np.where(~df.columns.isin(default_df.columns))[0]
            if len(indxmissing) > 0:
                raise ValueError(f'df does not contain column {default_df.columns[indxmissing]}')
            self.update_pick(df)
        else:
            raise ValueError('df input is not pd.DataFrame')

    def new_row_frompick(self, sample=None, trace=None, amp=None, group=None):
        new_row = self.init_df(nrow=1)
        new_row['sample'] = sample
        new_row['trace'] = trace
        new_row['amp'] = amp
        new_row['group'] = group
        return new_row

    def add_spike(self, new_row):
        df = self.picks
        # Check columns of new row
        indxmissing = np.where(~df.columns.isin(new_row.columns))[0]
        if len(indxmissing) > 0:
            raise ValueError(f'new_row does not contain column {df.columns[indxmissing]}')
        # Append new row
        df_updated = pd.concat([df, new_row])
        df_updated = df_updated.reset_index(drop=True)
        self.update_pick(df_updated)

    def remove_spike(self, indx_remove):
        df = self.picks
        if df.shape[0] > 0 and len(indx_remove) > 0:  # Update only if non-empty
            df_updated = df.drop(indx_remove).copy()
            df_updated = df_updated.reset_index(drop=True)
            self.update_pick(df_updated)


    def indx_select(self, sample, trace, s_range=0.5 * 30000, tr_range=3):
        iclose = np.where(np.logical_and(
            np.abs(self.picks['sample'] - sample) <= (s_range + 1),
            np.abs(self.picks['trace'] - trace) <= (tr_range + 1)
        ))[0]
        return iclose

class EphysViewer(EasyQC):
    keyPressed = QtCore.pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ctrl.model.pickspikes = PickSpikes()
        self.menufile.setEnabled(True)
        self.settings = QtCore.QSettings('int-brain-lab', 'EphysViewer')
        self.header_curves = {}
        # menus handling
        # menu pick
        self.menupick = self.menuBar().addMenu('&Pick')
        self.action_pick = QtWidgets.QAction('Pick', self)
        self.action_pick.setCheckable(True)
        self.menupick.addAction(self.action_pick)
        self.action_pick.triggered.connect(self.menu_pick_callback)
        # menu channels
        self.action_label_channels = QtWidgets.QAction('Label channels', self)
        self.action_label_channels.setCheckable(True)
        self.menupick.addAction(self.action_label_channels)
        # finish init
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
            self.keyPressed.connect(self.on_key_picking_mode)
        else:
            self.viewBox_seismic.scene().sigMouseClicked.disconnect(self.mouseClickPickingEvent)
            self.keyPressed.disconnect(self.on_key_picking_mode)

    def keyPressEvent(self, event):
        super(EphysViewer, self).keyPressEvent(event)
        self.keyPressed.emit(event.key())

    def on_key_picking_mode(self, key):
        """
        When the pick action is enabled this is triggered on key press
        """
        match key:
            case QtCore.Qt.Key.Key_Space:
                self.ctrl.model.pick_group += 1

    def mouseClickPickingEvent(self, event):
        """
        When the pick action is enabled this is triggered on mouse click
        - left button click sets a point
        - shift + left button removes a point
        - control + left does not wrap on maximum around pick
        - space increments the group number
        """

        if event.buttons() == QtCore.Qt.RightButton:
            self.ctrl.model.pickspikes.pick_group += 1  # TODO check logic of incrementing here
        if event.buttons() != QtCore.Qt.LeftButton:
            return
        TR_RANGE = 3
        S_RANGE = int(0.5 / self.ctrl.model.si)
        qxy = self.imageItem_seismic.mapFromScene(event.scenePos())
        s, tr = (qxy.x(), qxy.y())
        # if event.buttons() == QtCore.Qt.MiddleButton:
        match event.modifiers():  # upon clicking:
            # --- Remove a spike when shift key is pressed
            case QtCore.Qt.KeyboardModifier.ShiftModifier:
                i_remv = self.ctrl.model.pickspikes.indx_select(
                    sample=s, trace=tr, s_range=S_RANGE, tr_range=TR_RANGE)
                self.ctrl.model.pickspikes.remove_spike(i_remv)
                tmax = None

            # --- Add a spike
            case QtCore.Qt.ControlModifier:
                # the control modifier prevents wrapping around the nearby maximal voltage
                tmax, xmax = (int(round(s)), int(round(tr)))

            case _:
                # if no key is pressed and click, automatic wrapping around the nearby maximal voltage
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

        if tmax is not None:  # When spike is added
            # we add the spike to the dataframe
            amp = self.ctrl.model.data[tmax, xmax]
            group = 0  # TODO group
            # Create new row
            new_row = self.ctrl.model.pickspikes.new_row_frompick(
                sample=tmax, trace=xmax, amp=amp, group=group)
            self.ctrl.model.pickspikes.add_spike(new_row=new_row)

        # updates scatter plot
        self.ctrl.add_scatter(self.ctrl.model.pickspikes.picks['sample'] * self.ctrl.model.si,
                              self.ctrl.model.pickspikes.picks['trace'],
                              label='_picks', rgb=PICK_COLOR)

    def save_current_plot(self, filename):
        """
        Saves only the currently shown plot to `filename`.
        :param filename:
        :return:
        """
        self.plotItem_seismic.grab().save(filename)


def viewephys(data, fs, channels=None, br=None, title='ephys', t0=0, t_scalar=T_SCALAR, a_scalar=A_SCALAR,
              colormap=None) -> EphysViewer:
    """
    :param data: [nc, ns]
    :param fs:
    :param channels: dictionary of trace headers (nc, ) or dataframe (nc, ncolumns)
    :param br:
    :param title:
    :param colormap: non-standard colormap from colorcet or matplotlib such as "PuOr"
    :return:
    """

    easyqc.qt.create_app()
    ev = EphysViewer._get_or_create(title=title)

    if channels is None:
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
    if colormap is not None:
        ev.setColorMap(colormap)
    return ev
