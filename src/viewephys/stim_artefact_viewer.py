from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from neuropixel import trace_header
from qtpy import QtCore, QtGui, QtWidgets

from viewephys.gui import A_SCALAR, T_SCALAR, EphysViewer, create_app

if TYPE_CHECKING:
    import matplotlib


def samples_to_seconds(samples, fs: float, first_time: float = 0.0):
    """Convert sample indices to seconds on the recording's time axis.

    The recording may start at a non-zero time, so ``first_time`` (the
    timestamp of sample 0) is added to the sample offset. Accepts scalars,
    arrays or pandas Series.
    """
    return samples / fs + first_time


def seconds_to_samples(seconds, fs: float, first_time: float = 0.0):
    """Inverse of :func:`samples_to_seconds`, rounded to the nearest sample.

    Returns a Python ``int`` for scalar input and an integer array otherwise.
    """
    samples = np.round((np.asarray(seconds, dtype=float) - first_time) * fs)
    samples = samples.astype(int)
    return samples.item() if samples.ndim == 0 else samples


class SelectableLinearRegionItem(pg.LinearRegionItem):
    sigRegionClicked = QtCore.Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for line in self.lines:
            line.sigClicked.connect(self._emit_clicked)

    def _emit_clicked(self, *_) -> None:
        self.sigRegionClicked.emit()

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.sigRegionClicked.emit()
            ev.accept()
            return
        super().mouseClickEvent(ev)


class StimArtefactViewer(EphysViewer):
    request_jump_to_time = QtCore.Signal(float)

    def __init__(
        self, events_path: str | Path, fs, first_time: float = 0.0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fs = fs
        # Timestamp of sample 0; the events file stores sample indices, which
        # map onto the recording time axis as ``sample / fs + first_time``.
        self.first_time = first_time
        self._context_menu_region_center_time: float | None = None
        self._init_region_context_menu()
        self.extend_gui()
        self._event_regions: list[SelectableLinearRegionItem] = []
        self._visible_event_indices: list[int] = []
        self.selected_event_idx: int | None = 0
        self._regions_hidden = False
        self.regions_changed_since_last_save = 0

        # do some validation
        self.events_path = events_path
        self.events = pd.read_csv(events_path)
        self._populate_region_select()
        self._update_event_count_label()
        self._fill_region_controls_lineedits()

        # For undo/redo, for now we just take snapshots of the
        # events dataframe. This is very poor performance but
        # a) these should not be too big
        # b) saving deltas is very fiddly because add / delete event compeltely changes
        # all indicies
        self.events_snapshots = []
        self.redo_snapshots = []
        # Seed the stack with the baseline so the first edit is undoable and
        # the current state always matches the top of events_snapshots.
        self.update_snapshot()

        # Yellow overlay of the stim-removed signal, drawn on top of the
        # wiggle traces. It shares the primary model's geometry, gain and
        # channel selection (see _refresh_overlay).
        self._overlay_data = None
        self._overlay_item = pg.PlotDataItem(visible=False)
        self._overlay_item.setPen(pg.mkPen("#FFD700", width=0.9))
        self.plotItem_seismic.addItem(self._overlay_item)
        self.checkBox_stim_overlay.toggled.connect(self._on_overlay_toggled)
        # Overlay only applies in wiggle mode; the viewer boots in density mode.
        self._refresh_overlay()

    def _get_event_times(self, ev_idx: int) -> tuple[float, float]:
        event = self.events.loc[ev_idx]
        start_time = self._samples_to_seconds(float(event["start_sample"]))
        stop_time = self._samples_to_seconds(float(event["end_sample"]))
        return start_time, stop_time

    def _samples_to_seconds(self, samples):
        return samples_to_seconds(samples, self.fs, self.first_time)

    def _seconds_to_samples(self, seconds):
        return seconds_to_samples(seconds, self.fs, self.first_time)

    def _init_region_context_menu(self) -> None:
        self.viewBox_seismic.menu.addSeparator()
        action = QtWidgets.QAction("Add Region", self.viewBox_seismic.menu)
        action.triggered.connect(self._on_add_region_menu_clicked)
        self.viewBox_seismic.menu.addAction(action)

    def mouseClick(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            point = self.viewBox_seismic.mapSceneToView(event.scenePos())
            self._context_menu_region_center_time = float(point.x())
        super().mouseClick(event)

    def _on_add_region_menu_clicked(self) -> None:
        self._on_add_event_clicked(self._context_menu_region_center_time)

    def _on_region_clicked(self, ev_idx: int) -> None:
        if ev_idx == self.selected_event_idx:
            return
        self.move_to_region(ev_idx)

    def update_snapshot(self):
        self.events_snapshots.append(
            {
                "events": self.events.copy(),
                "selected_event_idx": self.selected_event_idx,
            }
        )
        if len(self.events_snapshots) > 20:
            self.events_snapshots.pop(0)
        # A fresh action invalidates any redo branch.
        self.redo_snapshots.clear()

    @staticmethod
    def _format_time(value: float) -> str:
        return f"{value:.6f}"

    def _set_region_controls_times(self, start_time: float, stop_time: float) -> None:
        for line_edit, value in (
            (self.lineEdit_stim_t0, start_time),
            (self.lineEdit_stim_t1, stop_time),
        ):
            line_edit.blockSignals(True)
            line_edit.setText(self._format_time(value))
            line_edit.blockSignals(False)

    def _fill_region_controls_lineedits(self):
        start_time, stop_time = self._get_event_times(self.selected_event_idx)
        self._set_region_controls_times(start_time, stop_time)

    def _apply_selected_event_time_change(
        self, start_time: float, stop_time: float, region: bool, region_controls: bool
    ) -> bool:
        ev_idx = self.selected_event_idx
        if ev_idx is None:
            return False

        if start_time >= stop_time:
            self.update_widgets_after_time_change(*self._get_event_times(ev_idx))
            return False

        start_sample = self._seconds_to_samples(start_time)
        end_sample = self._seconds_to_samples(stop_time)
        if start_sample >= end_sample:
            self.update_widgets_after_time_change(*self._get_event_times(ev_idx))
            return False

        self.events.loc[ev_idx, "start_sample"] = start_sample
        self.events.loc[ev_idx, "end_sample"] = end_sample
        self.update_widgets_after_time_change(
            self._samples_to_seconds(start_sample),
            self._samples_to_seconds(end_sample),
            region=region,
            region_controls=region_controls,
        )
        return True

    def update_widgets_after_time_change(
        self, start_time, stop_time, region=True, region_controls=True
    ):
        if region:
            if self.selected_event_idx in self._visible_event_indices:
                current_region = self.get_visible_region(self.selected_event_idx)
                current_region.setRegion((start_time, stop_time))

        if region_controls:
            self._set_region_controls_times(start_time, stop_time)

        ev_idx = self.selected_event_idx
        if ev_idx is None:
            return

        combo_idx = int(self.events.index.get_loc(ev_idx))
        self.comboBox_stim_event_index.setItemText(
            combo_idx, self._format_event_combo_text(ev_idx)
        )

    def _format_event_combo_text(self, ev_idx: int) -> str:
        start_time, stop_time = self._get_event_times(ev_idx)
        return (
            f"{ev_idx} ({self._format_time(start_time)}, "
            f"{self._format_time(stop_time)})"
        )

    def _on_region_change_finished(self, ev_idx: int) -> None:
        if ev_idx != self.selected_event_idx:
            return
        start_time, stop_time = self.get_visible_region(ev_idx).getRegion()
        self._apply_selected_event_time_change(
            float(start_time), float(stop_time), region=False, region_controls=True
        )
        self.update_snapshot()

    def _on_region_lineedits_changed(self) -> None:
        try:
            start_time = float(self.lineEdit_stim_t0.text())
            stop_time = float(self.lineEdit_stim_t1.text())
        except ValueError:
            # TODO: flesh this out with proper exceptiosn
            self._fill_region_controls_lineedits()
            return

        self._apply_selected_event_time_change(
            start_time, stop_time, region=True, region_controls=False
        )
        self.update_snapshot()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:
        if event is None:
            return

        key = event.key()
        if key == QtCore.Qt.Key.Key_Left:
            self._on_prev_event_clicked()
            event.accept()
            return
        if key == QtCore.Qt.Key.Key_Right:
            self._on_next_event_clicked()
            event.accept()
            return

        super().keyPressEvent(event)

    def _on_undo_clicked(self) -> None:
        # Top of the stack is the current state; need a prior state to step to.
        if len(self.events_snapshots) <= 1:
            return
        # Move the current state onto redo, then restore the previous one
        # (now the last entry in the list).
        recent_snapshot = self.events_snapshots.pop()
        selected_region_idx = recent_snapshot["selected_event_idx"]
        self.redo_snapshots.append(recent_snapshot)
        snapshot = self.events_snapshots[-1]
        self.events = snapshot["events"].copy()
        self.selected_event_idx = selected_region_idx
        self._repopulate_event_widgets()
        self.move_to_region(self.selected_event_idx)

    def _on_redo_clicked(self) -> None:
        if not self.redo_snapshots:
            return
        snapshot = self.redo_snapshots.pop()
        self.events_snapshots.append(snapshot)
        self.events = snapshot["events"].copy()
        self.selected_event_idx = snapshot["selected_event_idx"]
        self._repopulate_event_widgets()
        self.move_to_region(self.selected_event_idx)

    def _set_save_status(self, message: str, success: bool = True) -> None:
        if success:
            self.label_stim_save_status.setStyleSheet("color: #3c78b8;")
        else:
            self.label_stim_save_status.setStyleSheet("color: #a61b1b;")
        self.label_stim_save_status.setText(message)
        self.label_stim_save_status.setVisible(True)

    def _update_region_change_status(self, *, visible: bool | None = None) -> None:
        self.label_stim_region_change_status.setStyleSheet("color: #3c78b8;")
        self.label_stim_region_change_status.setText(
            f"Events since saved: {self.regions_changed_since_last_save}"
        )
        if visible is not None:
            self.label_stim_region_change_status.setVisible(visible)

    def _on_save_clicked(self) -> None:
        destination = Path(self.events_path)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            previous_mtime_ns = None
            if destination.exists():
                previous_mtime_ns = destination.stat().st_mtime_ns

            self.events.to_csv(destination, index=False)

            deadline = datetime.now().timestamp() + 2.0
            while True:
                if destination.exists() and destination.stat().st_size > 0:
                    current_mtime_ns = destination.stat().st_mtime_ns
                    if (
                        previous_mtime_ns is None
                        or current_mtime_ns > previous_mtime_ns
                    ):
                        break
                if datetime.now().timestamp() >= deadline:
                    raise OSError("Saved file timestamp did not update")
                sleep(0.01)

            self._set_save_status(f"File Saved ({datetime.now().strftime('%H:%M:%S')})")
            self.regions_changed_since_last_save = 0
            self._update_region_change_status(visible=True)
        except Exception as exc:
            self._set_save_status("Save failed", success=False)
            raise RuntimeError(f"Failed to save events: {exc}") from exc

    def _on_load_clicked(self) -> None:
        self.events = pd.read_csv(self.events_path)
        self.selected_event_idx = int(self.events.index[0])
        self.regions_changed_since_last_save = 0
        self._repopulate_event_widgets()
        self._update_region_change_status(visible=True)
        self.update_snapshot()

    def _get_event_insert_position(self, start_sample: int) -> int:
        """Return the index where an event should be inserted by time order."""
        for ev_idx, row in self.events.iterrows():
            if float(row["start_sample"]) > start_sample:
                return int(ev_idx)
        return len(self.events)

    def _on_add_event_clicked(self, center_time: float | None = None) -> None:
        # Placing the event in the current view needs model timing (set with data).
        if self.model.data is None:
            return

        xmin, xmax = self.viewBox_seismic.viewRange()[0]
        if isinstance(center_time, bool):
            center_time = None
        mid_time = (xmin + xmax) / 2 if center_time is None else float(center_time)
        half_width = (xmax - xmin) * 0.01

        start_sample = self._seconds_to_samples(mid_time - half_width)
        end_sample = self._seconds_to_samples(mid_time + half_width)
        if end_sample <= start_sample:
            end_sample = start_sample + 1

        insert_pos = self._get_event_insert_position(start_sample)

        new_row = pd.DataFrame(
            [{"start_sample": start_sample, "end_sample": end_sample}]
        )
        # ignore_index rebuilds a clean RangeIndex so label == position holds.
        self.events = pd.concat(
            [
                self.events.iloc[:insert_pos],
                new_row,
                self.events.iloc[insert_pos:],
            ],
            ignore_index=True,
        )
        self.selected_event_idx = insert_pos
        # plot_events_as_regions() (via _repopulate_event_widgets) rebuilds the
        # visible regions list from the reindexed events.
        self.regions_changed_since_last_save += 1
        self._repopulate_event_widgets()
        self._update_region_change_status(visible=True)
        self.update_snapshot()

    def _on_del_event_clicked(self) -> None:
        if self.events.shape[0] == 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Cannot delete event",
                "You cannot delete all events.",
            )
            return

        current = self.selected_event_idx
        self.events = self.events.drop(index=current).reset_index(drop=True)
        self.selected_event_idx = 0 if current == 0 else current - 1
        self.regions_changed_since_last_save -= 1
        self._repopulate_event_widgets()
        self._update_region_change_status(visible=True)
        self.update_snapshot()

    def _update_event_count_label(self) -> None:
        self.groupBox_region_select.setTitle(
            f"Region Select ({len(self.events)} events)"
        )

    def _repopulate_event_widgets(self) -> None:
        self.plot_events_as_regions()
        self._populate_region_select()
        self._set_event_combo_index(self.selected_event_idx)
        self._fill_region_controls_lineedits()
        self._update_event_count_label()

    # can centralise this somwehow?
    @staticmethod
    def _get_or_create(
        fs: float,
        events_path: str | Path,
        title=None,
        first_time: float = 0.0,
    ) -> EphysViewer:
        ev = next(
            filter(
                lambda e: e.isVisible() and e.windowTitle() == title,
                StimArtefactViewer._instances(),
            ),
            None,
        )
        if ev is None:
            ev = StimArtefactViewer(
                fs=fs, events_path=events_path, first_time=first_time
            )  # maybe set this like data?
            ev.setWindowTitle(title)

        return ev

    # TODO: will need to handle the events list going empty.
    def _populate_region_select(self) -> None:
        self.comboBox_stim_event_index.blockSignals(True)
        self.comboBox_stim_event_index.clear()
        self.comboBox_stim_event_index.addItems(
            [
                self._format_event_combo_text(ev_idx)
                for ev_idx, row in self.events.iterrows()
            ]
        )
        self.comboBox_stim_event_index.setCurrentIndex(0)
        self.comboBox_stim_event_index.blockSignals(False)

    def _set_event_combo_index(self, ev_idx: int | None) -> None:
        self.comboBox_stim_event_index.blockSignals(True)
        if ev_idx is None:
            try:
                self.comboBox_stim_event_index.setCurrentIndex(-1)
            finally:
                self.comboBox_stim_event_index.blockSignals(False)
            return
        try:
            combo_idx = int(self.events.index.get_loc(ev_idx))
            self.comboBox_stim_event_index.setCurrentIndex(combo_idx)
        finally:
            self.comboBox_stim_event_index.blockSignals(False)

    # TODO: this should be two functions
    def plot_events_as_regions(self) -> None:
        for region in self._event_regions:
            self.plotItem_seismic.removeItem(region)
        self._event_regions.clear()
        self._visible_event_indices.clear()

        view_start = float(self.model.t0)
        view_stop = float(self.model.t0 + self.model.ns * self.model.si)

        # The events file stores stim samples; convert to seconds for the time axis.
        start_s = self._samples_to_seconds(self.events["start_sample"])
        stop_s = self._samples_to_seconds(self.events["end_sample"])
        visible = self.events[(start_s >= view_start) & (stop_s <= view_stop)]

        if visible.empty:
            return

        for event_idx, event in visible.iterrows():
            region = SelectableLinearRegionItem(
                values=(
                    self._samples_to_seconds(float(event["start_sample"])),
                    self._samples_to_seconds(float(event["end_sample"])),
                ),
                orientation="vertical",
                movable=False,
            )
            region.sigRegionClicked.connect(
                lambda ev_idx=int(event_idx): self._on_region_clicked(ev_idx)
            )
            region.sigRegionChangeFinished.connect(
                lambda *_, ev_idx=int(event_idx): self._on_region_change_finished(
                    ev_idx
                )
            )
            self.set_region_inactive(region)
            self.plotItem_seismic.addItem(region)
            self._event_regions.append(region)
            self._visible_event_indices.append(int(event_idx))

        self._set_event_combo_index(self.selected_event_idx)
        # The selected event may be outside the current view (e.g. after load /
        # delete); only activate its region when it is actually plotted.
        if self.selected_event_idx in self._visible_event_indices:
            self.set_region_active(self.get_visible_region(self.selected_event_idx))

    def move_to_region(self, new_region_idx: int) -> None:
        if new_region_idx not in self._visible_event_indices:
            event = self.events.loc[new_region_idx]
            midpoint = self._samples_to_seconds(
                (float(event["start_sample"]) + float(event["end_sample"])) / 2
            )

            self._set_event_combo_index(new_region_idx)
            self.selected_event_idx = new_region_idx
            self._fill_region_controls_lineedits()
            self.request_jump_to_time.emit(midpoint)
            return

        current_region = self.get_visible_region(self.selected_event_idx)
        if current_region is not None:
            self.set_region_inactive(current_region)

        new_region = self.get_visible_region(new_region_idx)
        assert new_region is not None
        self.set_region_active(new_region)

        self._set_event_combo_index(new_region_idx)
        self.selected_event_idx = new_region_idx
        self._fill_region_controls_lineedits()

    def get_visible_region(self, ev_idx):
        if ev_idx not in self._visible_event_indices:
            return None
        region_idx = self._visible_event_indices.index(ev_idx)
        return self._event_regions[region_idx]

    def set_region_active(self, region):
        region.setMovable(True)
        region.setBrush(pg.mkBrush(40, 120, 255, 95))
        region.setHoverBrush(pg.mkBrush(40, 120, 255, 125))
        region.update()

    def set_region_inactive(self, region):
        region.setMovable(False)
        region.setBrush(pg.mkBrush(150, 150, 150, 80))
        region.setHoverBrush(pg.mkBrush(150, 150, 150, 105))
        region.update()

    def _on_event_combo_changed(self, combo_idx: int) -> None:
        if combo_idx < 0:
            return
        ev_idx = int(self.events.index[combo_idx])
        self.move_to_region(ev_idx)

    def _on_prev_event_clicked(self) -> None:
        new_idx = self.selected_event_idx - 1
        if new_idx < 0:
            return
        self.move_to_region(new_idx)

    def _on_next_event_clicked(self) -> None:
        new_idx = self.selected_event_idx + 1
        if new_idx >= self.events.shape[0]:
            return
        self.move_to_region(new_idx)

    def _on_hide_regions_toggled(self, checked: bool) -> None:
        """Show or hide the event regions overlaid on the plot."""
        self._regions_hidden = bool(checked)
        for region in self._event_regions:
            region.setVisible(not self._regions_hidden)

    def set_overlay_data(self, data) -> None:
        """Store the stim-removed array used for the wiggle overlay.

        It must share the primary model's geometry (same shape as the array
        passed to the model after the .T transform).
        """
        self._overlay_data = data
        self._refresh_overlay()

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        # The overlay reuses the wiggle formula, so it is only meaningful in
        # wiggle mode: grey out the toggle when density mode is active.
        is_wiggle = self.ctrl is self._ctrl_wiggle
        self.checkBox_stim_overlay.setEnabled(is_wiggle)
        show = (
            self.checkBox_stim_overlay.isChecked()
            and self._overlay_data is not None
            and is_wiggle
        )
        if not show:
            self._overlay_item.setVisible(False)
            self._overlay_item.clear()
            return
        # Same gain / trace selection / auto-spacing as the primary trace.
        x, y = self._ctrl_wiggle.compute_wiggle_xy(self._overlay_data)
        self._overlay_item.setData(x=x, y=y)
        self._overlay_item.setVisible(True)

    def set_display_mode(self, mode: str) -> None:
        super().set_display_mode(mode)
        self._refresh_overlay()

    def editGain(self) -> None:
        super().editGain()
        self._refresh_overlay()

    def on_checkbox_auto_space_wiggle(self, value: bool) -> None:
        super().on_checkbox_auto_space_wiggle(value)
        self._refresh_overlay()

    def _on_show_trace_header_toggled(self, checked: bool) -> None:
        """Show or hide the trace header plots (top, right) and trace combobox."""
        visible = bool(checked)
        self.plotItem_header_h.setVisible(visible)
        self.plotItem_header_v.setVisible(visible)
        # frame_header_h contains the trace (header) combobox.
        self.frame_header_h.setVisible(visible)

    def init_trace_header(self, trace_header: dict) -> None:
        assert "ids" in trace_header
        if self.listWidget_stim_channels.count() == 0:
            # only populate on the first data update, not every refresh
            self._populate_channel_list(trace_header["ids"])

    def _populate_channel_list(self, ids) -> None:
        self.listWidget_stim_channels.clear()
        for trace_idx, channel_id in enumerate(ids):
            item = QtWidgets.QListWidgetItem(str(channel_id))
            item.setData(QtCore.Qt.UserRole, trace_idx)
            self.listWidget_stim_channels.addItem(item)
        self._update_channel_count_label()

    def _update_channel_count_label(self) -> None:
        total = self.listWidget_stim_channels.count()
        chosen = len(self.listWidget_stim_channels.selectedItems())
        self.label_stim_channel_count.setText(f"{chosen} / {total} channels")

    def _on_channels_all(self) -> None:
        self.listWidget_stim_channels.selectAll()
        self._update_channel_count_label()

    def _on_channels_none(self) -> None:
        self.listWidget_stim_channels.clearSelection()
        self._update_channel_count_label()

    def _on_channels_group_toggled(self, checked: bool) -> None:
        """Collapse the channels group box content when unchecked."""
        self._channels_content.setVisible(checked)

    def _on_channels_apply(self) -> None:
        items = self.listWidget_stim_channels.selectedItems()
        total = self.listWidget_stim_channels.count()
        if not items or len(items) == total:
            trace_indices = np.arange(self.model.ntr)
            self.listWidget_stim_channels.selectAll()
        else:
            trace_indices = np.sort(
                np.array(
                    [int(item.data(QtCore.Qt.UserRole)) for item in items], dtype=int
                )
            )

        for ctrl in (self._ctrl_image, self._ctrl_wiggle):
            ctrl.trace_indices = trace_indices
        # Reset the view bounds so they match the filtered number of traces.
        t0, si, ns = self.model.t0, self.model.si, self.model.ns
        x0 = self.model.x0
        tlim = [t0, t0 + ns * si]
        clim = [x0 - 0.5, x0 + trace_indices.size - 0.5]
        ctrls = [self._ctrl_image, self._ctrl_wiggle]
        ctrls.remove(self.ctrl)
        ctrls.append(self.ctrl)
        for ctrl in ctrls:
            ctrl._update_plotItem(tlim=tlim, clim=clim)
            ctrl.set_header()
            ctrl.set_gain()
            self._update_channel_count_label()
        # Channel filtering changed trace_indices; keep the overlay in step.
        self._refresh_overlay()

    def extend_gui(self) -> None:
        # Add a View-menu toggle for the trace header plots and trace combobox.
        self.actionShowTraceHeader = QtWidgets.QAction("Show trace header", self)
        self.actionShowTraceHeader.setObjectName("actionShowTraceHeader")
        self.actionShowTraceHeader.setCheckable(True)
        self.actionShowTraceHeader.setChecked(True)
        self.actionShowTraceHeader.toggled.connect(self._on_show_trace_header_toggled)
        self.menuView.addAction(self.actionShowTraceHeader)

        # Bottom container.
        bottom_widget = QtWidgets.QWidget(self.centralwidget)
        bottom_widget.setObjectName("widget_stim_bottom")
        bottom_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        bottom_outer = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_outer.setContentsMargins(0, 4, 0, 0)
        bottom_outer.setSpacing(2)

        self.checkBox_stim_overlay = QtWidgets.QCheckBox(
            "Overlay stim removed", bottom_widget
        )
        self.checkBox_stim_overlay.setObjectName("checkBox_stim_overlay")
        self.checkBox_stim_overlay.setToolTip(
            "Overlay the stim-artefact-removed signal in yellow on top of "
            "the wiggle traces."
        )
        self.pushButton_stim_overlay = self.checkBox_stim_overlay
        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.setContentsMargins(8, 0, 8, 0)
        overlay_row.setSpacing(8)
        overlay_row.addWidget(self.checkBox_stim_overlay)
        overlay_row.addStretch(1)
        bottom_outer.addLayout(overlay_row)

        self.checkBox_hide_regions = QtWidgets.QCheckBox("Hide regions", bottom_widget)
        self.checkBox_hide_regions.setObjectName("checkBox_hide_regions")
        self.checkBox_hide_regions.setToolTip(
            "Hide the event regions overlaid on the plot."
        )
        self.checkBox_hide_regions.toggled.connect(self._on_hide_regions_toggled)
        hide_regions_row = QtWidgets.QHBoxLayout()
        hide_regions_row.setContentsMargins(8, 0, 8, 0)
        hide_regions_row.setSpacing(8)
        hide_regions_row.addWidget(self.checkBox_hide_regions)
        hide_regions_row.addStretch(1)
        bottom_outer.addLayout(hide_regions_row)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_outer.addLayout(bottom_layout)

        # Region time controls.
        self.groupBox_region_controls = QtWidgets.QGroupBox(
            "Region Controls", bottom_widget
        )
        self.groupBox_region_controls.setObjectName("groupBox_region_controls")
        region_controls_layout = QtWidgets.QVBoxLayout(self.groupBox_region_controls)
        region_controls_layout.setContentsMargins(8, 8, 8, 8)
        region_controls_layout.setSpacing(4)

        time_validator = QtGui.QDoubleValidator(
            0.0, 1e12, 6, self.groupBox_region_controls
        )
        time_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        time_validator.setLocale(QtCore.QLocale.c())

        edits_row = QtWidgets.QHBoxLayout()
        edits_row.setSpacing(8)
        self.label_stim_t0 = QtWidgets.QLabel("t0", self.groupBox_region_controls)
        self.lineEdit_stim_t0 = QtWidgets.QLineEdit(self.groupBox_region_controls)
        self.lineEdit_stim_t0.setObjectName("lineEdit_stim_t0")
        self.lineEdit_stim_t0.setMinimumWidth(110)
        self.lineEdit_stim_t0.setValidator(time_validator)
        self.lineEdit_stim_t0.editingFinished.connect(self._on_region_lineedits_changed)
        self.label_stim_t1 = QtWidgets.QLabel("t1", self.groupBox_region_controls)
        self.lineEdit_stim_t1 = QtWidgets.QLineEdit(self.groupBox_region_controls)
        self.lineEdit_stim_t1.setObjectName("lineEdit_stim_t1")
        self.lineEdit_stim_t1.setMinimumWidth(110)
        self.lineEdit_stim_t1.setValidator(time_validator)
        self.lineEdit_stim_t1.editingFinished.connect(self._on_region_lineedits_changed)
        edits_row.addWidget(self.label_stim_t0)
        edits_row.addWidget(self.lineEdit_stim_t0)
        edits_row.addWidget(self.label_stim_t1)
        edits_row.addWidget(self.lineEdit_stim_t1)
        edits_row.addStretch(1)

        btn_row = QtWidgets.QGridLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setHorizontalSpacing(4)
        btn_row.setVerticalSpacing(2)
        self.pushButton_stim_undo = QtWidgets.QPushButton(
            "Undo", self.groupBox_region_controls
        )
        self.pushButton_stim_undo.setObjectName("pushButton_stim_undo")
        self.pushButton_stim_undo.clicked.connect(self._on_undo_clicked)
        self.pushButton_stim_redo = QtWidgets.QPushButton(
            "Redo", self.groupBox_region_controls
        )
        self.pushButton_stim_redo.setObjectName("pushButton_stim_redo")
        self.pushButton_stim_redo.clicked.connect(self._on_redo_clicked)
        self.pushButton_stim_save = QtWidgets.QPushButton(
            "Save", self.groupBox_region_controls
        )
        self.pushButton_stim_save.setObjectName("pushButton_stim_save")
        self.pushButton_stim_save.clicked.connect(self._on_save_clicked)
        self.pushButton_stim_load = QtWidgets.QPushButton(
            "Load", self.groupBox_region_controls
        )
        self.pushButton_stim_load.setObjectName("pushButton_stim_load")
        self.pushButton_stim_load.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.pushButton_stim_undo, 0, 0)
        btn_row.addWidget(self.pushButton_stim_redo, 0, 1)
        btn_row.addWidget(self.pushButton_stim_save, 0, 2)
        btn_row.addWidget(self.pushButton_stim_load, 0, 3)
        btn_row.setColumnStretch(4, 1)

        self.label_stim_save_status = QtWidgets.QLabel(
            "", self.groupBox_region_controls
        )
        self.label_stim_save_status.setObjectName("label_stim_save_status")
        self.label_stim_save_status.setVisible(False)
        self.label_stim_save_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stim_save_status.setWordWrap(True)
        self.label_stim_save_status.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.label_stim_save_status.setStyleSheet("color: #3c78b8;")
        btn_row.addWidget(self.label_stim_save_status, 1, 2, 1, 2)

        self.label_stim_region_change_status = QtWidgets.QLabel(
            "", self.groupBox_region_controls
        )
        self.label_stim_region_change_status.setObjectName(
            "label_stim_region_change_status"
        )
        self.label_stim_region_change_status.setVisible(False)
        self.label_stim_region_change_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stim_region_change_status.setWordWrap(True)
        self.label_stim_region_change_status.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        btn_row.addWidget(self.label_stim_region_change_status, 2, 2, 1, 2)

        region_controls_layout.addLayout(edits_row)
        region_controls_layout.addLayout(btn_row)
        region_controls_layout.addStretch(1)

        # Event selection controls.
        self.groupBox_region_select = QtWidgets.QGroupBox(
            "Region Select", bottom_widget
        )
        self.groupBox_region_select.setObjectName("groupBox_region_select")
        self.groupBox_region_select.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        event_select_layout = QtWidgets.QVBoxLayout(self.groupBox_region_select)
        event_select_layout.setContentsMargins(8, 8, 8, 8)
        event_select_layout.setSpacing(4)

        self.label_stim_event_index = QtWidgets.QLabel(
            "Evt. index (label)", self.groupBox_region_select
        )
        self.comboBox_stim_event_index = QtWidgets.QComboBox(
            self.groupBox_region_select
        )
        self.comboBox_stim_event_index.setObjectName("comboBox_stim_event_index")
        self.comboBox_stim_event_index.setMinimumWidth(140)
        self.comboBox_stim_event_index.setEditable(True)
        self.comboBox_stim_event_index.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.comboBox_stim_event_index.setToolTip(
            "Type an event index and press Enter to jump to it. The "
            "prev / next arrows step by the configured step size."
        )
        self.comboBox_stim_event_index.activated.connect(self._on_event_combo_changed)
        nav_widget = QtWidgets.QWidget(self.groupBox_region_select)
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        self.pushButton_stim_prev = QtWidgets.QPushButton("<", nav_widget)
        self.pushButton_stim_prev.setObjectName("pushButton_stim_prev")
        self.pushButton_stim_prev.setMaximumWidth(36)
        self.pushButton_stim_prev.clicked.connect(self._on_prev_event_clicked)
        self.lineEdit_stim_step = QtWidgets.QLineEdit("1", nav_widget)
        self.lineEdit_stim_step.setObjectName("lineEdit_stim_step")
        self.lineEdit_stim_step.setAlignment(QtCore.Qt.AlignCenter)
        step_validator = QtGui.QIntValidator(1, 10_000_000, self.lineEdit_stim_step)
        self.lineEdit_stim_step.setValidator(step_validator)
        self.pushButton_stim_next = QtWidgets.QPushButton(">", nav_widget)
        self.pushButton_stim_next.setObjectName("pushButton_stim_next")
        self.pushButton_stim_next.setMaximumWidth(36)
        self.pushButton_stim_next.clicked.connect(self._on_next_event_clicked)
        nav_layout.addWidget(self.pushButton_stim_prev)
        nav_layout.addWidget(self.lineEdit_stim_step)
        nav_layout.addWidget(self.pushButton_stim_next)

        action_widget = QtWidgets.QWidget(self.groupBox_region_select)
        action_layout = QtWidgets.QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self.pushButton_stim_add = QtWidgets.QPushButton("Add", action_widget)
        self.pushButton_stim_add.setObjectName("pushButton_stim_add")
        self.pushButton_stim_add.clicked.connect(self._on_add_event_clicked)
        self.pushButton_stim_del = QtWidgets.QPushButton("Del", action_widget)
        self.pushButton_stim_del.setObjectName("pushButton_stim_del")
        self.pushButton_stim_del.clicked.connect(self._on_del_event_clicked)
        action_layout.addWidget(self.pushButton_stim_add)
        action_layout.addWidget(self.pushButton_stim_del)

        event_select_layout.addWidget(self.label_stim_event_index)
        event_select_layout.addWidget(self.comboBox_stim_event_index)
        event_select_layout.addWidget(nav_widget)
        event_select_layout.addWidget(action_widget)

        # Channel selection controls.
        self.groupBox_channels = QtWidgets.QGroupBox("Channels", self.centralwidget)
        self.groupBox_channels.setObjectName("groupBox_channels")
        self.groupBox_channels.setCheckable(True)
        self.groupBox_channels.setChecked(True)
        self.groupBox_channels.toggled.connect(self._on_channels_group_toggled)

        channels_layout = QtWidgets.QVBoxLayout(self.groupBox_channels)
        channels_layout.setContentsMargins(8, 8, 8, 8)
        channels_layout.setSpacing(4)

        content = QtWidgets.QWidget(self.groupBox_channels)
        content.setObjectName("widget_stim_channels_content")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)

        self.label_stim_channel_count = QtWidgets.QLabel("0 / 0 channels", content)
        self.label_stim_channel_count.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stim_channel_count.setObjectName("label_stim_channel_count")

        self.listWidget_stim_channels = QtWidgets.QListWidget(content)
        self.listWidget_stim_channels.setObjectName("listWidget_stim_channels")
        self.listWidget_stim_channels.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.listWidget_stim_channels.setMinimumWidth(120)
        self.listWidget_stim_channels.setMinimumHeight(120)

        btn_widget = QtWidgets.QWidget(content)
        btn_layout = QtWidgets.QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)
        self.pushButton_stim_channels_all = QtWidgets.QPushButton("All", btn_widget)
        self.pushButton_stim_channels_all.setObjectName("pushButton_stim_channels_all")
        self.pushButton_stim_channels_all.clicked.connect(self._on_channels_all)
        self.pushButton_stim_channels_none = QtWidgets.QPushButton("None", btn_widget)
        self.pushButton_stim_channels_none.setObjectName(
            "pushButton_stim_channels_none"
        )
        self.pushButton_stim_channels_none.clicked.connect(self._on_channels_none)
        self.pushButton_stim_channels_apply = QtWidgets.QPushButton("Apply", btn_widget)
        self.pushButton_stim_channels_apply.setObjectName(
            "pushButton_stim_channels_apply"
        )
        self.pushButton_stim_channels_apply.clicked.connect(self._on_channels_apply)
        btn_layout.addWidget(self.pushButton_stim_channels_all)
        btn_layout.addWidget(self.pushButton_stim_channels_none)
        btn_layout.addWidget(self.pushButton_stim_channels_apply)

        content_layout.addWidget(self.label_stim_channel_count)
        content_layout.addWidget(self.listWidget_stim_channels, 1)
        content_layout.addWidget(btn_widget)

        channels_layout.addWidget(content)
        self._channels_content = content
        self._channels_btn_widget = btn_widget

        # Main layout placement.
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.groupBox_region_controls, 0)
        bottom_layout.addWidget(self.groupBox_region_select, 0)
        bottom_layout.addStretch(1)

        self.gridLayout_4.addWidget(self.groupBox_channels, 0, 1, 1, 1)
        self.gridLayout_4.addWidget(bottom_widget, 1, 0, 1, 2)
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 0)
        self.gridLayout_4.setRowStretch(0, 1)
        self.gridLayout_4.setRowStretch(1, 0)


def stim_artefact_viewer(
    data: np.ndarray,
    fs: float,
    channels: dict | None = None,
    br=None,
    events_path: str | Path | None = None,
    title: str = "ephys",
    t0: float = 0.0,
    first_time: float = 0.0,
    t_scalar: float = T_SCALAR,
    a_scalar: float = A_SCALAR,
    colormap: str | pg.ColorMap | matplotlib.colors.Colormap | None = None,
    data_stim_removed: np.ndarray | None = None,
) -> StimArtefactViewer:
    create_app()
    # we need the fs upfront here to convert samples in events to time
    ev = StimArtefactViewer._get_or_create(
        fs=fs, events_path=events_path, title=title, first_time=first_time
    )

    if channels is None:
        channels = trace_header(version=1)

    if isinstance(channels, dict):
        if "ids" not in channels:
            n_channels = (
                data.shape[0]
                if data is not None
                else len(channels.get("receiver_line", []))
            )
            channels = {**channels, "ids": np.arange(n_channels)}
    else:
        if "ids" not in channels:
            n_channels = data.shape[0] if data is not None else len(channels)
            channels = channels.copy()
            channels["ids"] = np.arange(n_channels)

    if data is not None:
        if ev.fs is not None:
            assert fs == ev.fs
        ev.model.set_data(data.T * a_scalar, si=1 / fs, header=channels, t0=t0, taxis=0)
        ev.ctrl.set_model()
        ev.init_trace_header(channels)  # populate channel list on first data set
        ev._fill_region_controls_lineedits()
        ev.plot_events_as_regions()  # TODO: centralise
        ev._on_channels_apply()  # TODO: fold channel filtering into set_model.
        if data_stim_removed is not None:
            # Same geometry as the primary data; apply the matching transform.
            assert data_stim_removed.shape == data.shape
            ev.set_overlay_data(data_stim_removed.T * a_scalar)

    ev.show()
    if colormap is not None:
        ev.setColorMap(colormap)

    return ev
