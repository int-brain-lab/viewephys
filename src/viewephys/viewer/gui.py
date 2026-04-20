import abc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from . import qt
from .pgtools import ImShowSpectrogram
from .view import Ui_MainWindow

PARAMS_TRACE_PLOTS = {
    "neighbors": 2,
    "color": pg.mkColor((31, 119, 180)),
}

DISPLAY_MODE_DENSITY = "density"
DISPLAY_MODE_WIGGLE = "wiggle"


@dataclass
class Model:
    """Class for keeping track of the visualized data."""

    data: np.ndarray
    header: np.ndarray
    si: float = 1.0
    nx: int = 1
    ny: int = 1
    t0: float = 0
    x0: float = 0
    taxis: int = 0

    def auto_gain(self) -> float:
        rmsnan = np.nansum(self.data**2, axis=self.taxis) / np.sum(
            ~np.isnan(self.data), axis=self.taxis
        )
        return 20 * np.log10(np.median(np.sqrt(rmsnan)))

    def get_trace_spectrogram(self, c, trange=None):
        from scipy.signal import spectrogram

        tr = self.get_trace(c, trange=trange)
        fscale, tscale, tf = spectrogram(
            tr, fs=1 / self.si, nperseg=50, nfft=512, window="cosine", noverlap=48
        )
        tscale += trange[0]
        tf = 20 * np.log10(tf + np.finfo(float).eps)
        return fscale, tscale, tf

    def get_trace_spectrum(self, c, trange=None, neighbors=0):
        tr = self.get_trace(c, trange=trange, neighbors=neighbors)
        psd = 20 * np.log10(np.abs(np.fft.rfft(tr)) - np.finfo(float).eps)
        return np.fft.rfftfreq(tr.size, self.si), psd

    def get_trace(self, c, trange=None, neighbors=0):
        """
        Get trace according to index, taking into account the orientation of the model.
        """
        trsel = np.arange(-neighbors, neighbors + 1) + int(np.floor(c))
        trsel = trsel[np.logical_and(trsel < self.ntr, trsel >= 0)]
        if trange is not None:
            first_s = int((trange[0] - self.t0) / self.si)
            last_s = int((trange[1] - self.t0) / self.si)
            sl = slice(first_s, last_s)
        else:
            sl = slice(None)
        if self.caxis == 0:
            return np.squeeze(self.data[trsel, sl].T)
        return np.squeeze(self.data[sl, trsel])

    def set_data(self, data, header=None, si=None, t0=0, x0=0, taxis=1):
        assert (header is not None) or si
        if data.ndim >= 3:
            data = np.reshape(data, (-1, data.shape[-1]))
        self.x0 = x0
        self.t0 = t0
        self.header = header
        self.data = data
        self.taxis = taxis
        self.nx, self.ny = self.data.shape
        if self.taxis == 1:
            self.ntr, self.ns = self.data.shape
            self.caxis = 0
        else:
            self.ns, self.ntr = self.data.shape
            self.caxis = 1
        if si is not None:
            self.si = si
        elif isinstance(header, float):
            self.si = header
        else:
            self.si = header.si[0]
        self.header = {"trace": np.arange(self.ntr)}
        if header is not None:
            self.header.update(header)

    @property
    def tscale(self):
        return np.arange(self.ns) * self.si + self.t0


class EasyQC(QtWidgets.QMainWindow, Ui_MainWindow):
    """Reusable seismic-style viewer used by viewephys."""

    model: Model
    layers = None
    QT_APP = None

    @staticmethod
    def _instances():
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = qt.create_app()
        return [w for w in app.topLevelWidgets() if isinstance(w, EasyQC)]

    @staticmethod
    def _get_or_create(title=None):
        eqc = next(
            filter(
                lambda e: e.isVisible() and e.windowTitle() == title,
                EasyQC._instances(),
            ),
            None,
        )
        if eqc is None:
            eqc = EasyQC()
            eqc.setWindowTitle(title)
        return eqc

    @property
    def ctrl(self):
        return (
            self._ctrl_wiggle
            if self._display_mode == DISPLAY_MODE_WIGGLE
            else self._ctrl_image
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = {}
        self.model = Model(None, None)
        self._display_mode = DISPLAY_MODE_DENSITY
        self._ctrl_image = ControllerImage(self)
        self._ctrl_wiggle = ControllerWiggle(self)
        icon_path = Path(__file__).resolve().parent.parent.joinpath("viewephys.svg")
        self.setWindowIcon(QtGui.QIcon(str(icon_path)))
        background_color = self.palette().color(self.backgroundRole())
        self.plotItem_seismic.setAspectLocked(False)
        self.imageItem_seismic = pg.ImageItem()
        self.plotItem_seismic.setBackground(background_color)
        self.plotItem_seismic.addItem(self.imageItem_seismic)
        self.plotDataItem_wiggle = pg.PlotDataItem(visible=False)
        self.plotDataItem_wiggle.setPen(pg.mkPen("#ebc000"))
        self.plotItem_seismic.addItem(self.plotDataItem_wiggle)
        self.viewBox_seismic = self.plotItem_seismic.getPlotItem().getViewBox()
        self._init_menu()
        self._init_cmenu()
        self.plotDataItem_header_h = pg.PlotDataItem()
        self.plotItem_header_h.addItem(self.plotDataItem_header_h)
        self.plotItem_seismic.setXLink(self.plotItem_header_h)
        self.plotDataItem_header_v = pg.PlotDataItem()
        self.plotItem_header_h.setBackground(background_color)
        self.plotItem_header_v.addItem(self.plotDataItem_header_v)
        self.plotItem_header_v.setBackground(background_color)
        self.plotItem_seismic.setYLink(self.plotItem_header_v)
        ax = self.plotItem_seismic.getAxis("left")
        ax.setStyle(
            tickTextWidth=60, autoReduceTextSpace=False, autoExpandTextSpace=False
        )
        ax = self.plotItem_header_h.getAxis("left")
        ax.setStyle(
            tickTextWidth=60, autoReduceTextSpace=False, autoExpandTextSpace=False
        )
        ax = self.plotItem_header_v.getAxis("left")
        ax.setStyle(showValues=False)
        # Use one authoritative time axis on the main plot.
        bottom_axis = self.plotItem_seismic.getAxis("bottom")
        bottom_axis.setLabel("Time", units="s")
        bottom_axis.enableAutoSIPrefix(True)

        # Keep the top strip x-linked, but hide its duplicate x-axis.
        self.plotItem_header_h.getPlotItem().hideAxis("bottom")

        self.hoverPlotWidgets = {"Trace": None, "Spectrum": None, "Spectrogram": None}
        scene = self.viewBox_seismic.scene()
        self.proxy = pg.SignalProxy(
            scene.sigMouseMoved, rateLimit=60, slot=self.mouseMoveEvent
        )
        scene.sigMouseClicked.connect(self.mouseClick)
        self.lineEdit_gain.returnPressed.connect(self.editGain)
        self.lineEdit_sort.returnPressed.connect(self.editSort)
        self.comboBox_header.activated[str].connect(self.ctrl.set_header)
        self.viewBox_seismic.sigRangeChanged.connect(self.on_sigRangeChanged)
        self.horizontalScrollBar.valueChanged.connect(self.on_horizontalSliderChange)
        self.verticalScrollBar.valueChanged.connect(self.on_verticalSliderChange)
        self.radio_density.toggled.connect(
            lambda checked: checked and self.set_display_mode(DISPLAY_MODE_DENSITY)
        )
        self.radio_wiggle.toggled.connect(
            lambda checked: checked and self.set_display_mode(DISPLAY_MODE_WIGGLE)
        )

    def _init_menu(self):
        self.actionColormap_CET_D6.triggered.connect(lambda: self.setColorMap("CET-D6"))
        self.actionColormap_CET_D1.triggered.connect(lambda: self.setColorMap("CET-D1"))
        self.actionColormap_CET_L2.triggered.connect(lambda: self.setColorMap("CET-L2"))
        self.actionColormap_MPL_PuOr.triggered.connect(lambda: self.setColorMap("PuOr"))

    def _init_cmenu(self):
        self.viewBox_seismic.scene().contextMenu = None
        self.plotItem_seismic.plotItem.ctrlMenu = None
        for action in self.viewBox_seismic.menu.actions():
            if action.text() == "View All":
                continue
            self.viewBox_seismic.menu.removeAction(action)
        self.viewBox_seismic.menu.addSeparator()
        action = QtWidgets.QAction("View Trace", self.viewBox_seismic.menu)
        action.triggered.connect(self.cmenu_ViewTrace)
        self.viewBox_seismic.menu.addAction(action)
        action = QtWidgets.QAction("View Spectrum", self.viewBox_seismic.menu)
        action.triggered.connect(self.cmenu_ViewSpectrum)
        self.viewBox_seismic.menu.addAction(action)
        action = QtWidgets.QAction("View Spectrogram", self.viewBox_seismic.menu)
        action.triggered.connect(self.cmenu_ViewSpectrogram)
        self.viewBox_seismic.menu.addAction(action)

    def keyPressEvent(self, event):
        key, modifiers = (event.key(), event.modifiers())
        if key == QtCore.Qt.Key_PageUp or (
            modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_A
        ):
            self.ctrl.set_gain(self.ctrl.gain - 3)
        elif key == QtCore.Qt.Key_PageDown or (
            modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_Z
        ):
            self.ctrl.set_gain(self.ctrl.gain + 3)
        elif modifiers == QtCore.Qt.ShiftModifier and key == QtCore.Qt.Key_P:
            self.ctrl.propagate(explode=True)
        elif modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_P:
            self.ctrl.propagate()
        elif key in (
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Down,
        ):
            self.translate_seismic(key, modifiers == QtCore.Qt.ControlModifier)
        elif modifiers == QtCore.Qt.ControlModifier and key == QtCore.Qt.Key_S:
            qt_app = QtWidgets.QApplication.instance()
            qt_app.clipboard().setPixmap(self.plotItem_seismic.grab())

    def editGain(self):
        self.ctrl.set_gain()

    def editSort(self, redraw=True):
        keys = self.lineEdit_sort.text().split(" ")
        self.ctrl.sort(keys, redraw=redraw)

    def mouseClick(self, event):
        if not event.double():
            return
        qxy = self.imageItem_seismic.mapFromScene(event.scenePos())
        tr, sample = (qxy.x(), qxy.y())
        print(tr, sample)

    def mouseMoveEvent(self, scene_pos):
        if isinstance(scene_pos, tuple):
            scene_pos = scene_pos[0]
        else:
            return
        if self.ctrl.model.data is None:
            return
        qpoint = self.imageItem_seismic.mapFromScene(scene_pos)
        c, t, a, h = self.ctrl.cursor2timetraceamp(qpoint)
        self.label_x.setText(f"{c:.0f}")
        self.label_t.setText(f"{t:.4f}")
        self.label_amp.setText(f"{a:2.2E}")
        htxt = h if isinstance(h, str) else f"{h:.4f}"
        self.label_h.setText(htxt)
        for key in self.hoverPlotWidgets:
            if (
                self.hoverPlotWidgets[key] is not None
                and self.hoverPlotWidgets[key].isVisible()
            ):
                self.ctrl.update_hover(qpoint, key)

    def translate_seismic(self, key, control_modifier):
        """Resize horizontal/vertical ranges via the keyboard."""
        rect = self.viewBox_seismic.viewRect()
        xlim, ylim = self.ctrl.limits()
        factor = 1 / 2 if control_modifier else 1 / 7
        dy = factor * rect.height()
        dx = factor * rect.width()
        if key == QtCore.Qt.Key_Down:
            yr = np.array([rect.y(), rect.y() + rect.height()]) + dy
            yr += np.min([0, ylim[1] - yr[1]])
            self.viewBox_seismic.setYRange(yr[0], yr[1], padding=0)
        elif key == QtCore.Qt.Key_Left:
            xr = np.array([rect.x(), rect.x() + rect.width()]) - dx
            xr += np.max([0, xlim[0] - xr[0]])
            self.viewBox_seismic.setXRange(xr[0], xr[1], padding=0)
        elif key == QtCore.Qt.Key_Right:
            xr = np.array([rect.x(), rect.x() + rect.width()]) + dx
            xr += np.min([0, xlim[1] - xr[1]])
            self.viewBox_seismic.setXRange(xr[0], xr[1], padding=0)
        elif key == QtCore.Qt.Key_Up:
            yr = np.array([rect.y(), rect.y() + rect.height()]) - dy
            yr += np.max([0, ylim[0] - yr[0]])
            self.viewBox_seismic.setYRange(yr[0], yr[1], padding=0)

    def on_sigRangeChanged(self, _):
        def set_scroll(scrollbar, current_range, bounds):
            current_span = current_range[1] - current_range[0]
            doclength = bounds[1] - bounds[0]
            if doclength <= 0:
                blocker = QtCore.QSignalBlocker(scrollbar)
                try:
                    scrollbar.setMaximum(0)
                    scrollbar.setPageStep(1)
                    scrollbar.setSingleStep(1)
                    scrollbar.setValue(0)
                finally:
                    del blocker
                return

            maximum = int((doclength - current_span) / doclength * 65536)
            page_step = max(1, 65536 - maximum)
            single_step = max(1, page_step // 10)
            value = int((current_range[0] - bounds[0]) / doclength * 65536)
            value = max(0, min(maximum, value))

            # Keep plot -> scrollbar synchronization from re-triggering
            # scrollbar -> plot updates.
            blocker = QtCore.QSignalBlocker(scrollbar)
            try:
                scrollbar.setMaximum(maximum)
                scrollbar.setPageStep(page_step)
                scrollbar.setSingleStep(single_step)
                scrollbar.setValue(value)
            finally:
                del blocker

        xr, yr = self.viewBox_seismic.viewRange()
        xl, yl = self.ctrl.limits()
        set_scroll(self.horizontalScrollBar, xr, xl)
        set_scroll(self.verticalScrollBar, yr, yl)

    def on_horizontalSliderChange(self, _):
        bounds = self.ctrl.limits()[0]
        current_range = self.viewBox_seismic.viewRange()[0]
        x = (
            float(self.horizontalScrollBar.value()) / 65536 * (bounds[1] - bounds[0])
            + bounds[0]
        )
        self.viewBox_seismic.setXRange(
            x, x + current_range[1] - current_range[0], padding=0
        )

    def on_verticalSliderChange(self, _):
        bounds = self.ctrl.limits()[1]
        current_range = self.viewBox_seismic.viewRange()[1]
        y = (
            float(self.verticalScrollBar.value()) / 65536 * (bounds[1] - bounds[0])
            + bounds[0]
        )
        self.viewBox_seismic.setYRange(
            y, y + current_range[1] - current_range[0], padding=0
        )

    def _cmenu_hover(self, key, image=False):
        if self.hoverPlotWidgets[key] is None:
            if image and key == "Spectrogram":
                self.hoverPlotWidgets[key] = ImShowSpectrogram()
            else:
                self.hoverPlotWidgets[key] = pg.plot(
                    [0], [0], pen=pg.mkPen(color=[180, 180, 180]), connect="finite"
                )
                self.hoverPlotWidgets[key].addItem(
                    pg.PlotCurveItem(
                        [0],
                        [0],
                        pen=pg.mkPen(color=PARAMS_TRACE_PLOTS["color"], width=1),
                        connect="finite",
                    )
                )
                self.hoverPlotWidgets[key].addItem(
                    pg.PlotDataItem(
                        [0],
                        [0],
                        symbolPen=pg.mkPen(color=[255, 0, 0]),
                        symbolSize=7,
                        symbol="star",
                    )
                )
                self.hoverPlotWidgets[key].setBackground(pg.mkColor("#ffffff"))
        self.hoverPlotWidgets[key].setVisible(True)

    def cmenu_ViewTrace(self):
        self._cmenu_hover("Trace")

    def cmenu_ViewSpectrum(self):
        self._cmenu_hover("Spectrum")

    def cmenu_ViewSpectrogram(self):
        self._cmenu_hover("Spectrogram", image=True)

    def setColorMap(self, cmap):
        if isinstance(cmap, str) and cmap not in pg.colormap.listMaps():
            cmap = pg.colormap.getFromMatplotlib(cmap)
        self.imageItem_seismic.setColorMap(cmap)

    def set_display_mode(self, mode: str) -> None:
        if mode == self._display_mode:
            return
        self._display_mode = mode
        if mode == DISPLAY_MODE_DENSITY:
            self.imageItem_seismic.setVisible(True)
            self.plotDataItem_wiggle.setVisible(False)
            self.plotDataItem_wiggle.clear()
            self.plotItem_seismic.setBackground("#000000")
        elif mode == DISPLAY_MODE_WIGGLE:
            self.imageItem_seismic.clear()
            self.imageItem_seismic.setVisible(False)
            self.plotDataItem_wiggle.setVisible(True)
            self.plotItem_seismic.setBackground("#193600")
        self.ctrl.set_model(reset_viewbox=False)


class Controller(abc.ABC):
    def __init__(self, view):
        self.view = view
        self.order = None
        self.transform = np.eye(3)
        self.trace_indices = None
        self.hkey = None

    @abc.abstractmethod
    def set_gain(self, gain=None):
        pass

    @abc.abstractmethod
    def _update_plotItem(self, tlim, clim):
        pass

    @property
    def model(self) -> Model:
        return self.view.model

    def set_model(self, reset_viewbox=True):
        t0, x0 = self.model.t0, self.model.x0
        self.remove_all_layers()
        self.trace_indices = np.arange(self.model.ntr)
        self.view.editSort(redraw=False)
        self._update_plotItem(
            tlim=[t0, t0 + self.model.ns * self.model.si] if reset_viewbox else None,
            clim=[x0 - 0.5, x0 + self.model.ntr - 0.5] if reset_viewbox else None,
        )
        if isinstance(self.model.header, dict):
            self.view.comboBox_header.clear()
            for hname in self.model.header:
                self.view.comboBox_header.addItem(hname)
        self.set_gain(gain=self.gain)
        self.set_header()

    def remove_all_layers(self):
        layers_dict = self.view.layers.copy()
        for label in layers_dict:
            self.remove_layer_from_label(label)

    def remove_layer_from_label(self, label):
        current_layer = self.view.layers.get(label)
        if current_layer is not None:
            current_layer["layer"].clear()
            self.view.plotItem_seismic.removeItem(current_layer["layer"])
            self.view.layers.pop(label)

    def _add_plotitem(
        self,
        plot_item_class,
        x,
        y,
        rgb=None,
        label="default",
        brush=None,
        pen=None,
        **kwargs,
    ):
        rgb = (0, 255, 0) if rgb is None else rgb
        self.remove_layer_from_label(label)
        new_scatter = plot_item_class()
        self.view.layers[label] = {"layer": new_scatter, "type": "scatter"}
        self.view.plotItem_seismic.addItem(new_scatter)
        brush = pg.mkBrush(rgb) if brush is None else brush
        pen = pg.mkPen(rgb) if pen is None else pen
        new_scatter.setData(x=x, y=y, brush=brush, name=label, pen=pen, **kwargs)
        return new_scatter

    def add_curve(self, *args, **kwargs):
        return self._add_plotitem(pg.PlotCurveItem, *args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        return self._add_plotitem(pg.ScatterPlotItem, *args, **kwargs)

    def cursor2timetraceamp(self, qpoint):
        ixy = self.cursor2ind(qpoint)
        a = self.model.data[ixy[0], ixy[1]]
        xy_ = np.matmul(self.transform, np.array([qpoint.x(), qpoint.y(), 1]))
        t = xy_[self.model.taxis]
        c = ixy[self.model.caxis]
        h = self.model.header[self.hkey][ixy[self.model.caxis]]
        return c, t, a, h

    def cursor2ind(self, qpoint):
        ix = int(
            np.maximum(
                0, np.minimum(qpoint.x() - self.transform[1, 2], self.model.nx - 1)
            )
        )
        iy = int(
            np.maximum(
                0, np.minimum(qpoint.y() - self.transform[1, 1], self.model.ny - 1)
            )
        )
        return ix, iy

    def limits(self):
        ixlim = [0, self.model.nx]
        iylim = [0, self.model.ny]
        x, y, _ = np.matmul(self.transform, np.c_[ixlim, iylim, [1, 1]].T)
        return x, y

    def propagate(self, explode=False):
        eqcs = self.view._instances()
        if len(eqcs) == 1:
            return
        for i, eqc in enumerate(eqcs):
            if eqc is self.view:
                continue
            eqc.setColorMap(self.view.imageItem_seismic.getColorMap() or "CET-L2")
            eqc.setGeometry(self.view.geometry())
            eqc.ctrl.set_gain(self.gain)
            eqc.plotItem_seismic.setXLink(self.view.plotItem_seismic)
            eqc.plotItem_seismic.setYLink(self.view.plotItem_seismic)
            eqc.lineEdit_sort.setText(self.view.lineEdit_sort.text())
            eqc.ctrl.sort(eqc.lineEdit_sort.text())
            if explode and i % 2 == 1:
                rect = self.view.geometry()
                rect.translate(rect.width(), 0)
                eqc.setGeometry(rect)

    @property
    def gain(self):
        str_gain = self.view.lineEdit_gain.text()
        if str_gain.strip() == "":
            return self.model.auto_gain()
        return float(str_gain)

    def set_header(self):
        key = self.view.comboBox_header.currentText()
        if key not in self.model.header:
            return
        self.hkey = key
        traces = np.arange(self.trace_indices.size)
        values = self.model.header[self.hkey][self.trace_indices]
        if not np.issubdtype(values.dtype, np.number):
            return
        if self.model.taxis == 1:
            self.view.plotDataItem_header_h.setData(x=traces, y=values)
        elif self.model.taxis == 0:
            self.view.plotDataItem_header_v.setData(y=traces, x=values)

    def snapshot(self, file, xrange=None, yrange=None, gain=None, window_size=None):
        if yrange is not None:
            self.view.viewBox_seismic.setYRange(*yrange)
        if xrange is not None:
            self.view.viewBox_seismic.setXRange(*xrange)
        if gain is not None:
            self.set_gain(gain)
        if window_size is not None:
            self.view.resize(*window_size)
        self.view.grab().save(str(file))

    def sort(self, keys, redraw=True):
        if not (set(keys).issubset(set(self.model.header.keys()))):
            return
        if len(keys) == 0:
            return
        self.trace_indices = np.lexsort([self.model.header[k] for k in keys])
        if redraw:
            self.redraw()

    def update_hover(self, qpoint, key):
        c, t, a, _ = self.cursor2timetraceamp(qpoint)
        if key == "Trace":
            plotitem = self.view.hoverPlotWidgets[key].getPlotItem()
            traces = self.model.get_trace(c, neighbors=PARAMS_TRACE_PLOTS["neighbors"])
            nc = traces.shape[1]
            xdata = np.tile(np.r_[self.model.tscale, np.nan], nc).T
            ydata = np.r_[traces, np.ones((1, nc)) * np.nan].T
            plotitem.items[0].setData(xdata.flatten(), ydata.flatten())
            trace = self.model.get_trace(c)
            plotitem.items[1].setData(self.model.tscale, trace)
            plotitem.items[2].setData([t], [a])
            plotitem.setXRange(*self.trange)
        elif key == "Spectrum":
            plotitem = self.view.hoverPlotWidgets[key].getPlotItem()
            plotitem.items[0].setData(
                *self.model.get_trace_spectrum(c, trange=self.trange)
            )
        elif key == "Spectrogram":
            self.view.hoverPlotWidgets[key].set_data(
                self.model.get_trace(c), fs=1 / self.model.si
            )

    @property
    def tscale(self):
        return np.arange(self.model.ns) * self.model.si + self.model.t0

    @property
    def trange(self):
        return self.view.viewBox_seismic.viewRange()[self.model.taxis]

    @property
    def crange(self):
        return self.view.viewBox_seismic.viewRange()[self.model.caxis]


class ControllerWiggle(Controller):
    def _update_plotItem(self, tlim=None, clim=None):
        if self.model.taxis == 0:
            xlim, ylim = (tlim, clim)
            wiggle_y = np.r_[self.model.data, np.ones(self.model.ntr)[np.newaxis, :]]
            wiggle_y = (
                wiggle_y / (10 ** (self.gain / 20))
                + np.arange(self.model.ntr)[np.newaxis, :]
            )
            self.view.plotDataItem_wiggle.setData(
                x=np.tile(np.r_[self.tscale, np.nan], self.model.ntr),
                y=wiggle_y.T.flatten(),
            )
        elif self.model.taxis == 1:
            xlim, ylim = (clim, tlim)
        else:
            raise ValueError("taxis must be 0 (horizontal axis) or 1 (vertical axis)")
        if tlim is not None and clim is not None:
            self.view.plotItem_header_h.setLimits(xMin=xlim[0], xMax=xlim[1])
            self.view.plotItem_header_v.setLimits(yMin=ylim[0], yMax=ylim[1])
            self.view.plotItem_seismic.setLimits(
                xMin=xlim[0], xMax=xlim[1], yMin=ylim[0], yMax=ylim[1]
            )
            xlim, ylim = self.limits()
            self.view.viewBox_seismic.setXRange(*xlim, padding=0)
            self.view.viewBox_seismic.setYRange(*ylim, padding=0)

    def set_gain(self, gain=None):
        if gain is None:
            gain = self.gain
        self.view.lineEdit_gain.setText(f"{gain:.1f}")
        self._update_plotItem()


class ControllerImage(Controller):
    def _update_plotItem(self, tlim, clim):
        x0, t0, si = self.model.x0, self.model.t0, self.model.si
        if self.model.taxis == 0:
            xlim, ylim = (tlim, clim)
            transform = [si, 0.0, 0.0, 0.0, 1, 0.0, t0 - si / 2, x0 - 0.5, 1.0]
            self.view.imageItem_seismic.setImage(self.model.data[:, self.trace_indices])
        elif self.model.taxis == 1:
            xlim, ylim = (clim, tlim)
            transform = [1.0, 0.0, 0.0, 0.0, si, 0.0, x0 - 0.5, t0 - si / 2, 1.0]
            self.view.imageItem_seismic.setImage(self.model.data[self.trace_indices, :])
            self.view.plotItem_seismic.invertY()
        else:
            raise ValueError("taxis must be 0 (horizontal axis) or 1 (vertical axis)")
        if tlim is not None and clim is not None:
            self.transform = np.array(transform).reshape((3, 3)).T
            self.view.imageItem_seismic.setTransform(QtGui.QTransform(*transform))
            self.view.plotItem_header_h.setLimits(xMin=xlim[0], xMax=xlim[1])
            self.view.plotItem_header_v.setLimits(yMin=ylim[0], yMax=ylim[1])
            self.view.plotItem_seismic.setLimits(
                xMin=xlim[0], xMax=xlim[1], yMin=ylim[0], yMax=ylim[1]
            )
            xlim, ylim = self.limits()
            self.view.viewBox_seismic.setXRange(*xlim, padding=0)
            self.view.viewBox_seismic.setYRange(*ylim, padding=0)

    def redraw(self):
        if self.model.taxis == 1:
            self.view.imageItem_seismic.setImage(self.model.data[self.trace_indices, :])
        elif self.model.taxis == 0:
            self.view.imageItem_seismic.setImage(self.model.data[:, self.trace_indices])
        self.set_header()
        self.set_gain()

    def set_gain(self, gain=None):
        if gain is None:
            gain = self.gain
        levels = 10 ** (gain / 20) * 4 * np.array([-1, 1])
        self.view.imageItem_seismic.setLevels(levels)
        self.view.lineEdit_gain.setText(f"{gain:.1f}")


def viewseis(w=None, si=0.002, h=None, title=None, t0=0, x0=0, taxis=1):
    eqc = EasyQC._get_or_create(title=title)
    if w is not None:
        eqc.model.set_data(w, si=si, header=h, x0=x0, t0=t0, taxis=taxis)
        eqc.ctrl.set_model()
    eqc.show()
    return eqc


if __name__ == "__main__":
    eqc = viewseis(None)
    app = pg.Qt.mkQApp()
    sys.exit(app.exec_())
