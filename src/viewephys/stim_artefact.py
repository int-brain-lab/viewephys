from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyqtgraph as pg
from iblutil.numerical import ismember
from neuropixel import trace_header
from qtpy import QtCore, QtGui, QtWidgets

from viewephys.data_model import SpikeInterfaceDataModel
from viewephys.gui import A_SCALAR, NSAMP_CHUNK, T_SCALAR, EphysBinViewer, EphysViewer
from viewephys.viewer.gui import DISPLAY_MODE_WIGGLE, ControllerWiggle
from viewephys.viewer.qt import create_app

if TYPE_CHECKING:
    import matplotlib


# Visual styling for the LinearRegionItem overlays.
_REGION_BRUSH = pg.mkBrush(120, 120, 120, 60)
_REGION_HOVER_BRUSH = pg.mkBrush(120, 120, 120, 90)
_REGION_SELECTED_BRUSH = pg.mkBrush(0, 100, 255, 90)
_REGION_SELECTED_HOVER_BRUSH = pg.mkBrush(0, 100, 255, 130)
_REGION_PEN = pg.mkPen(80, 80, 80, width=1)
_REGION_SELECTED_PEN = pg.mkPen(0, 60, 200, width=2)


class StimArtefactViewer(EphysViewer):
    """Ephys viewer extended with stim-artefact region editing.

    The viewer owns the in-memory ``events`` ``DataFrame`` (columns
    ``start, stop``) and the CSV path. The CSV file is written only when the
    user clicks **Save**; **Load** re-reads it and discards in-memory edits.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.settings = QtCore.QSettings("int-brain-lab", "StimArtefactViewer")

        # Event state.
        self.events: pd.DataFrame = pd.DataFrame(columns=["start", "stop"])
        self.csv_path: Path | None = None
        self._bin_viewer: StimArtefactBinViewer | None = None
        self._current_event_idx: int | None = None

        # Undo / redo stacks of DataFrame snapshots (deep copies).
        self._undo_stack: list[pd.DataFrame] = []
        self._redo_stack: list[pd.DataFrame] = []

        # Visible region overlays keyed by event row index.
        self._region_items: dict[int, pg.LinearRegionItem] = {}

        # Channel selection state.
        # ``_selected_traces`` is the set of original trace indices the user
        # has chosen to display; ``None`` means "show every channel" (default).
        # ``_full_sorted_indices`` snapshots the controller's full ordered
        # trace list right after the bin viewer applies its sort, so we can
        # always re-derive the filtered indices in display order.
        self._selected_traces: set[int] | None = None
        self._full_sorted_indices: np.ndarray | None = None

        # Guards to suppress feedback loops between signals.
        self._updating_ui = False
        self._updating_region = False

        # Optional yellow overlay (e.g. ``stim_artefact_removed``) drawn on
        # top of the wiggle traces. ``_overlay_data`` matches the shape of
        # ``self.model.data`` (samples × traces).
        self._overlay_enabled: bool = False
        self._overlay_data: np.ndarray | None = None

        # Replace the wiggle controller with a subclass that knows how to
        # render the overlay aligned to the same baselines as the raw
        # traces.
        self._ctrl_wiggle = _StimControllerWiggle(self)

        # Yellow overlay line item, sits above the main wiggle pen.
        self.plotDataItem_overlay = pg.PlotDataItem(visible=False)
        self.plotDataItem_overlay.setPen(pg.mkPen("#ebc000"))
        self.plotItem_seismic.addItem(self.plotDataItem_overlay)

        self._init_stim_controls()
        self._connect_stim_signals()

    @staticmethod
    def _get_or_create(title=None) -> "StimArtefactViewer":
        ev = next(
            filter(
                lambda e: e.isVisible() and e.windowTitle() == title,
                StimArtefactViewer._instances(),
            ),
            None,
        )
        if ev is None or not isinstance(ev, StimArtefactViewer):
            ev = StimArtefactViewer()
            ev.setWindowTitle(title)
        return ev

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_stim_controls(self) -> None:
        """Build the stim-artefact controls.

        Three ``QGroupBox`` panels are placed side-by-side in a single bottom
        row beneath the plot so the plot takes the full available height:
        - ``Region Controls`` (left, stretches): t0/t1/Undo/Redo/Save/Load.
        - ``Region Select`` (middle, fixed-ish width): events count, evt.
          index combo, prev/step/next, Consume/Add/Del.
        - ``Channels`` (right, fixed-ish width): multi-select list of channels
          with All/None/Apply controls that filter the displayed traces.
        """
        bottom_widget = QtWidgets.QWidget(self.centralwidget)
        bottom_widget.setObjectName("widget_stim_bottom")
        bottom_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )
        bottom_outer = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_outer.setContentsMargins(0, 4, 0, 0)
        bottom_outer.setSpacing(2)

        # Overlay toggle (yellow ``stim_artefact_removed`` traces over wiggle).
        # Thin horizontal strip across the top of the bottom controls area.
        self.checkBox_stim_overlay = QtWidgets.QCheckBox(
            "Overlay stim removed", bottom_widget
        )
        self.checkBox_stim_overlay.setObjectName("checkBox_stim_overlay")
        self.checkBox_stim_overlay.setToolTip(
            "Overlay the stim-artefact-removed signal in yellow on top of "
            "the wiggle traces."
        )
        # Backwards-compat alias: existing tests / code may still reference
        # ``pushButton_stim_overlay``; the checkbox exposes the same
        # ``toggled`` signal and ``isChecked``/``setChecked`` API.
        self.pushButton_stim_overlay = self.checkBox_stim_overlay
        overlay_row = QtWidgets.QHBoxLayout()
        overlay_row.setContentsMargins(8, 0, 8, 0)
        overlay_row.setSpacing(8)
        overlay_row.addWidget(self.checkBox_stim_overlay)
        overlay_row.addStretch(1)
        bottom_outer.addLayout(overlay_row)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_outer.addLayout(bottom_layout)

        self.groupBox_region_controls = QtWidgets.QGroupBox(
            "Region Controls", bottom_widget
        )
        self.groupBox_region_controls.setObjectName("groupBox_region_controls")
        self._build_stim_time_controls(self.groupBox_region_controls)

        self.groupBox_region_select = QtWidgets.QGroupBox(
            "Region Select", bottom_widget
        )
        self.groupBox_region_select.setObjectName("groupBox_region_select")
        # Size to its natural (preferred) width and height so it doesn't
        # stretch across the full bottom row.
        self.groupBox_region_select.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        self._build_stim_event_controls(self.groupBox_region_select)

        self.groupBox_channels = QtWidgets.QGroupBox("Channels", self.centralwidget)
        self.groupBox_channels.setObjectName("groupBox_channels")
        # Make the panel collapsible: clicking the title toggles the inner
        # contents widget visibility, so the user can hide the channel
        # picker when they don't need it.
        self.groupBox_channels.setCheckable(True)
        self.groupBox_channels.setChecked(True)
        # Compact width when collapsed: just enough for the title checkbox.
        self._channels_collapsed_max_width = 24
        self._build_stim_channel_controls(self.groupBox_channels)
        # Cap the expanded width to the widest channel label so the panel
        # never eats half the window. Computed after the list is built.
        self._channels_expanded_max_width = self._compute_channels_max_width()
        self.groupBox_channels.setMaximumWidth(self._channels_expanded_max_width)
        self.groupBox_channels.toggled.connect(self._on_channels_collapsed_toggled)

        # Symmetric left/right stretches so the region groups sit centred
        # in the bottom strip instead of bunching to one side.
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.groupBox_region_controls, 0)
        bottom_layout.addWidget(self.groupBox_region_select, 0)
        bottom_layout.addStretch(1)

        # Plot grid sits at row 0 col 0; channels group sits at row 0 col 1
        # so it lives next to the plot. Bottom controls span both columns.
        self.gridLayout_4.addWidget(self.groupBox_channels, 0, 1, 1, 1)
        self.gridLayout_4.addWidget(bottom_widget, 1, 0, 1, 2)
        # Plot column gets all the horizontal stretch; channels column is
        # sized to the widget's max width.
        self.gridLayout_4.setColumnStretch(0, 1)
        self.gridLayout_4.setColumnStretch(1, 0)
        # Plot row gets all vertical stretch; bottom-controls row hugs its
        # natural height instead of expanding.
        self.gridLayout_4.setRowStretch(0, 1)
        self.gridLayout_4.setRowStretch(1, 0)

    def _build_stim_time_controls(self, parent: QtWidgets.QWidget) -> None:
        """t0/t1 line-edits with Undo/Redo/Save/Load on a row beneath."""
        outer = QtWidgets.QVBoxLayout(parent)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        time_validator = QtGui.QDoubleValidator(0.0, 1e12, 6, parent)
        time_validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        time_validator.setLocale(QtCore.QLocale.c())

        # Row 1: t0 / t1 line-edits.
        edits_row = QtWidgets.QHBoxLayout()
        edits_row.setSpacing(8)
        self.label_stim_t0 = QtWidgets.QLabel("t0", parent)
        self.lineEdit_stim_t0 = QtWidgets.QLineEdit(parent)
        self.lineEdit_stim_t0.setObjectName("lineEdit_stim_t0")
        self.lineEdit_stim_t0.setMinimumWidth(110)
        self.lineEdit_stim_t0.setValidator(time_validator)

        self.label_stim_t1 = QtWidgets.QLabel("t1", parent)
        self.lineEdit_stim_t1 = QtWidgets.QLineEdit(parent)
        self.lineEdit_stim_t1.setObjectName("lineEdit_stim_t1")
        self.lineEdit_stim_t1.setMinimumWidth(110)
        self.lineEdit_stim_t1.setValidator(time_validator)
        edits_row.addWidget(self.label_stim_t0)
        edits_row.addWidget(self.lineEdit_stim_t0)
        edits_row.addWidget(self.label_stim_t1)
        edits_row.addWidget(self.lineEdit_stim_t1)
        edits_row.addStretch(1)

        # Row 2: Undo / Redo / Save / Load buttons.
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)
        self.pushButton_stim_undo = QtWidgets.QPushButton("Undo", parent)
        self.pushButton_stim_undo.setObjectName("pushButton_stim_undo")
        self.pushButton_stim_redo = QtWidgets.QPushButton("Redo", parent)
        self.pushButton_stim_redo.setObjectName("pushButton_stim_redo")
        self.pushButton_stim_save = QtWidgets.QPushButton("Save", parent)
        self.pushButton_stim_save.setObjectName("pushButton_stim_save")
        self.pushButton_stim_load = QtWidgets.QPushButton("Load", parent)
        self.pushButton_stim_load.setObjectName("pushButton_stim_load")
        btn_row.addWidget(self.pushButton_stim_undo)
        btn_row.addWidget(self.pushButton_stim_redo)
        btn_row.addWidget(self.pushButton_stim_save)
        btn_row.addWidget(self.pushButton_stim_load)
        btn_row.addStretch(1)

        outer.addLayout(edits_row)
        outer.addLayout(btn_row)
        outer.addStretch(1)

    def _build_stim_event_controls(self, parent: QtWidgets.QWidget) -> None:
        """Event-navigation widgets inside the Region Select group."""
        vbox = QtWidgets.QVBoxLayout(parent)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        self.label_stim_event_index = QtWidgets.QLabel("Evt. index (label)", parent)
        self.comboBox_stim_event_index = QtWidgets.QComboBox(parent)
        self.comboBox_stim_event_index.setObjectName("comboBox_stim_event_index")
        self.comboBox_stim_event_index.setMinimumWidth(140)

        # prev / step / next row (â† [step] â†’)
        nav_widget = QtWidgets.QWidget(parent)
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        self.pushButton_stim_prev = QtWidgets.QPushButton("\u2190", nav_widget)
        self.pushButton_stim_prev.setObjectName("pushButton_stim_prev")
        self.pushButton_stim_prev.setMaximumWidth(36)
        self.lineEdit_stim_step = QtWidgets.QLineEdit("1", nav_widget)
        self.lineEdit_stim_step.setObjectName("lineEdit_stim_step")
        self.lineEdit_stim_step.setAlignment(QtCore.Qt.AlignCenter)
        step_validator = QtGui.QIntValidator(1, 10_000_000, self.lineEdit_stim_step)
        self.lineEdit_stim_step.setValidator(step_validator)
        self.pushButton_stim_next = QtWidgets.QPushButton("\u2192", nav_widget)
        self.pushButton_stim_next.setObjectName("pushButton_stim_next")
        self.pushButton_stim_next.setMaximumWidth(36)
        nav_layout.addWidget(self.pushButton_stim_prev)
        nav_layout.addWidget(self.lineEdit_stim_step)
        nav_layout.addWidget(self.pushButton_stim_next)

        # Consume / Add / Del on a single horizontal row.
        action_widget = QtWidgets.QWidget(parent)
        action_layout = QtWidgets.QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        self.pushButton_stim_consume = QtWidgets.QPushButton("Consume", action_widget)
        self.pushButton_stim_consume.setObjectName("pushButton_stim_consume")
        self.pushButton_stim_add = QtWidgets.QPushButton("Add", action_widget)
        self.pushButton_stim_add.setObjectName("pushButton_stim_add")
        self.pushButton_stim_del = QtWidgets.QPushButton("Del", action_widget)
        self.pushButton_stim_del.setObjectName("pushButton_stim_del")
        action_layout.addWidget(self.pushButton_stim_consume)
        action_layout.addWidget(self.pushButton_stim_add)
        action_layout.addWidget(self.pushButton_stim_del)

        vbox.addWidget(self.label_stim_event_index)
        vbox.addWidget(self.comboBox_stim_event_index)
        vbox.addWidget(nav_widget)
        vbox.addWidget(action_widget)

    def _build_stim_channel_controls(self, parent: QtWidgets.QWidget) -> None:
        """Channel multi-select list inside the Channels group.

        The list is populated from the controller's current sorted trace
        order and stores the original trace index in each item's ``UserRole``
        so selection survives sort changes and chunk reloads.
        """
        vbox = QtWidgets.QVBoxLayout(parent)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        # Wrap all inner widgets in a single content widget so the group
        # box (made checkable below) can collapse the panel to its header.
        content = QtWidgets.QWidget(parent)
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
        self.pushButton_stim_channels_all.setObjectName(
            "pushButton_stim_channels_all"
        )
        self.pushButton_stim_channels_none = QtWidgets.QPushButton("None", btn_widget)
        self.pushButton_stim_channels_none.setObjectName(
            "pushButton_stim_channels_none"
        )
        self.pushButton_stim_channels_apply = QtWidgets.QPushButton(
            "Apply", btn_widget
        )
        self.pushButton_stim_channels_apply.setObjectName(
            "pushButton_stim_channels_apply"
        )
        btn_layout.addWidget(self.pushButton_stim_channels_all)
        btn_layout.addWidget(self.pushButton_stim_channels_none)
        btn_layout.addWidget(self.pushButton_stim_channels_apply)

        content_layout.addWidget(self.label_stim_channel_count)
        content_layout.addWidget(self.listWidget_stim_channels, 1)
        content_layout.addWidget(btn_widget)

        vbox.addWidget(content)
        self._channels_content = content

    def _connect_stim_signals(self) -> None:
        self.comboBox_stim_event_index.currentIndexChanged.connect(
            self._on_combo_changed
        )
        self.pushButton_stim_prev.clicked.connect(self._on_prev)
        self.pushButton_stim_next.clicked.connect(self._on_next)
        self.pushButton_stim_add.clicked.connect(self._on_add)
        self.pushButton_stim_del.clicked.connect(self._on_del)
        self.pushButton_stim_consume.clicked.connect(self._on_consume)
        self.pushButton_stim_undo.clicked.connect(self._on_undo)
        self.pushButton_stim_redo.clicked.connect(self._on_redo)
        self.pushButton_stim_save.clicked.connect(self._on_save)
        self.pushButton_stim_load.clicked.connect(self._on_load)
        self.pushButton_stim_channels_all.clicked.connect(self._on_channels_all)
        self.pushButton_stim_channels_none.clicked.connect(self._on_channels_none)
        self.pushButton_stim_channels_apply.clicked.connect(self._on_channels_apply)
        self.checkBox_stim_overlay.toggled.connect(self._on_overlay_toggled)
        # Auto-space toggle calls ``ctrl.set_model(reset_viewbox=False)``
        # on the base class, which wipes ``trace_indices`` back to
        # ``arange(ntr)`` (after re-applying the sort via ``editSort``).
        # Re-apply our channel filter so the user's selection survives.
        self.checkBox_wiggle_autospace.toggled.connect(
            lambda _checked: self._on_wiggle_layout_changed()
        )
        self._refresh_undo_redo_enabled()
        # Wiggle-only widgets must be disabled in density mode.
        self._sync_wiggle_widgets_enabled()

    # ------------------------------------------------------------------
    # Public API used by ``StimArtefactBinViewer``
    # ------------------------------------------------------------------
    def attach_events(
        self,
        events: pd.DataFrame,
        csv_path: Path,
        bin_viewer: "StimArtefactBinViewer",
    ) -> None:
        """Attach the editable events table backing the regions."""
        self.csv_path = Path(csv_path)
        self._bin_viewer = bin_viewer
        self._set_events(events, record_history=False)
        # Default selection: first event if any.
        if len(self.events):
            self._current_event_idx = 0
            self._sync_combo_to_current()
            self._sync_time_fields_to_current()

    def refresh_regions(self) -> None:
        """Redraw the LinearRegionItem overlays for the current chunk."""
        self._clear_region_items()
        if self.events.empty:
            return
        xmin, xmax = self.ctrl.limits()[0]
        # Show every event that overlaps the current visible data window.
        # (Times in ``events`` are in seconds; plot units are seconds as well
        # because ``T_SCALAR == 1``.)
        starts = self.events["start"].to_numpy() * T_SCALAR
        stops = self.events["stop"].to_numpy() * T_SCALAR
        visible = (stops >= xmin) & (starts <= xmax)
        for idx in np.flatnonzero(visible):
            self._add_region_item(int(idx))
        self._apply_region_highlight()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _set_events(
        self,
        events: pd.DataFrame,
        record_history: bool = True,
    ) -> None:
        """Replace the events frame, optionally snapshotting history.

        Does *not* write to disk — use the Save button (or call
        :meth:`_save_csv` directly) to persist changes.
        """
        if record_history:
            self._undo_stack.append(self.events.copy())
            self._redo_stack.clear()
        # Normalize: drop NaNs, keep only start/stop, sort by start, reset index.
        df = events.loc[:, ["start", "stop"]].dropna()
        df = df.sort_values("start", kind="mergesort").reset_index(drop=True)
        self.events = df
        self._refresh_event_combo()
        self.refresh_regions()
        self._refresh_undo_redo_enabled()

    def _save_csv(self) -> None:
        if self.csv_path is None:
            return
        self.events.to_csv(self.csv_path, index=False)

    def _refresh_event_combo(self) -> None:
        self._updating_ui = True
        try:
            self.comboBox_stim_event_index.clear()
            self.comboBox_stim_event_index.addItems(
                [
                    f"{i} ({row.start:.6f}, {row.stop:.6f})"
                    for i, row in self.events.iterrows()
                ]
            )
            self.groupBox_region_select.setTitle(
                f"Region Select ({len(self.events)} events)"
            )
            # Reset/clip current selection.
            if self.events.empty:
                self._current_event_idx = None
            elif self._current_event_idx is None:
                self._current_event_idx = 0
            else:
                self._current_event_idx = min(
                    self._current_event_idx, len(self.events) - 1
                )
            self._sync_combo_to_current()
        finally:
            self._updating_ui = False
        self._sync_time_fields_to_current()

    def _sync_combo_to_current(self) -> None:
        if self._current_event_idx is None:
            return
        self._updating_ui = True
        try:
            self.comboBox_stim_event_index.setCurrentIndex(self._current_event_idx)
        finally:
            self._updating_ui = False

    def _sync_time_fields_to_current(self) -> None:
        if self._current_event_idx is None or self.events.empty:
            self.lineEdit_stim_t0.clear()
            self.lineEdit_stim_t1.clear()
            return
        row = self.events.iloc[self._current_event_idx]
        self.lineEdit_stim_t0.setText(f"{float(row['start']):.6f}")
        self.lineEdit_stim_t1.setText(f"{float(row['stop']):.6f}")

    def _refresh_undo_redo_enabled(self) -> None:
        self.pushButton_stim_undo.setEnabled(bool(self._undo_stack))
        self.pushButton_stim_redo.setEnabled(bool(self._redo_stack))

    # ------------------------------------------------------------------
    # Region overlay management
    # ------------------------------------------------------------------
    def _clear_region_items(self) -> None:
        for region in self._region_items.values():
            self.plotItem_seismic.removeItem(region)
        self._region_items.clear()

    def _add_region_item(self, idx: int) -> None:
        row = self.events.iloc[idx]
        region = pg.LinearRegionItem(
            values=(float(row["start"]) * T_SCALAR, float(row["stop"]) * T_SCALAR),
            orientation="vertical",
            brush=_REGION_BRUSH,
            hoverBrush=_REGION_HOVER_BRUSH,
            pen=_REGION_PEN,
            movable=False,
        )
        region.setZValue(10)
        # Only the currently-selected region is interactive: unselected
        # regions cannot be dragged, and clicking on them does NOT change
        # the active selection (the combo / prev / next controls own that).
        # Swallow mouse events so the region neither moves nor steals
        # selection on click.
        region.mouseClickEvent = lambda ev: ev.ignore()
        region.sigRegionChangeFinished.connect(
            lambda r, i=idx: self._on_region_changed(i, r)
        )
        self.plotItem_seismic.addItem(region)
        self._region_items[idx] = region

    def _apply_region_highlight(self) -> None:
        for idx, region in self._region_items.items():
            selected = idx == self._current_event_idx
            # Only the active region is movable; others are locked.
            region.setMovable(selected)
            if selected:
                region.setBrush(_REGION_SELECTED_BRUSH)
                region.setHoverBrush(_REGION_SELECTED_HOVER_BRUSH)
                # ``LinearRegionItem`` does not expose a pen setter that
                # propagates to its child ``InfiniteLine``s, so set those
                # directly to make the boundary lines visible.
                for line in region.lines:
                    line.setPen(_REGION_SELECTED_PEN)
            else:
                region.setBrush(_REGION_BRUSH)
                region.setHoverBrush(_REGION_HOVER_BRUSH)
                for line in region.lines:
                    line.setPen(_REGION_PEN)
            # ``LinearRegionItem.setBrush`` only stores the brush; it does
            # not schedule a repaint, so without this the fill colour
            # only refreshes on the next mouse event. Force an immediate
            # paint so the highlight follows the line colour change.
            region.update()

    def _on_region_changed(self, idx: int, region: pg.LinearRegionItem) -> None:
        if self._updating_region or idx not in self._region_items:
            return
        if idx >= len(self.events):
            return
        new_t0, new_t1 = region.getRegion()
        new_t0 = float(new_t0) / T_SCALAR
        new_t1 = float(new_t1) / T_SCALAR
        if new_t1 < new_t0:
            new_t0, new_t1 = new_t1, new_t0
        # Snapshot history before mutating.
        self._undo_stack.append(self.events.copy())
        self._redo_stack.clear()
        self.events.at[idx, "start"] = new_t0
        self.events.at[idx, "stop"] = new_t1
        # Resort (drag may reorder events); track which row was dragged.
        sorted_df = self.events.sort_values("start", kind="mergesort")
        new_pos = int(sorted_df.index.get_loc(idx))
        self.events = sorted_df.reset_index(drop=True)
        self._current_event_idx = new_pos
        self._refresh_event_combo()
        self.refresh_regions()
        self._refresh_undo_redo_enabled()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _scroll_to_current(self) -> None:
        if self._current_event_idx is None or self.events.empty:
            return
        if self._bin_viewer is None:
            return
        row = self.events.iloc[self._current_event_idx]
        center = (float(row["start"]) + float(row["stop"])) / 2.0
        # Only jump (which reloads the chunk and resets the view) when the
        # region center is outside the currently visible x-range. Switching
        # between regions that are already on screen should preserve the
        # user's zoom level instead of forcing a full re-zoom.
        try:
            xmin, xmax = self.viewBox_seismic.viewRange()[0]
            center_view = center * T_SCALAR
            if xmin <= center_view <= xmax:
                return
        except Exception:
            pass
        self._bin_viewer.jump_to_time(center)

    def _get_step(self) -> int:
        text = self.lineEdit_stim_step.text().strip()
        try:
            return max(1, int(text))
        except ValueError:
            return 1

    # ------------------------------------------------------------------
    # Slot implementations
    # ------------------------------------------------------------------
    def _on_combo_changed(self, idx: int) -> None:
        if self._updating_ui or idx < 0 or idx >= len(self.events):
            return
        self._current_event_idx = idx
        self._sync_time_fields_to_current()
        self._apply_region_highlight()
        self._scroll_to_current()

    def _on_prev(self) -> None:
        if self.events.empty or self._current_event_idx is None:
            return
        new_idx = max(0, self._current_event_idx - self._get_step())
        self._current_event_idx = new_idx
        self._sync_combo_to_current()
        self._sync_time_fields_to_current()
        self._apply_region_highlight()
        self._scroll_to_current()

    def _on_next(self) -> None:
        if self.events.empty or self._current_event_idx is None:
            return
        new_idx = min(len(self.events) - 1, self._current_event_idx + self._get_step())
        self._current_event_idx = new_idx
        self._sync_combo_to_current()
        self._sync_time_fields_to_current()
        self._apply_region_highlight()
        self._scroll_to_current()

    def _on_add(self) -> None:
        try:
            t0 = float(self.lineEdit_stim_t0.text())
            t1 = float(self.lineEdit_stim_t1.text())
        except ValueError:
            return
        if t1 < t0:
            t0, t1 = t1, t0
        new_row = pd.DataFrame({"start": [t0], "stop": [t1]})
        new_events = pd.concat([self.events, new_row], ignore_index=True)
        # Insertion position (after sorting) becomes the new current selection.
        sorted_df = new_events.sort_values("start", kind="mergesort").reset_index(
            drop=True
        )
        # Find the inserted row's position (first match on both start/stop).
        match = (sorted_df["start"] == t0) & (sorted_df["stop"] == t1)
        new_pos = int(np.flatnonzero(match.to_numpy())[0])
        self._current_event_idx = new_pos
        self._set_events(sorted_df, record_history=True)
        self._scroll_to_current()

    def _on_del(self) -> None:
        if self.events.empty or self._current_event_idx is None:
            return
        idx = self._current_event_idx
        new_events = self.events.drop(index=idx).reset_index(drop=True)
        # Keep selection in-bounds; prefer staying on the same position so the
        # user can step through deletions.
        if new_events.empty:
            self._current_event_idx = None
        else:
            self._current_event_idx = min(idx, len(new_events) - 1)
        self._set_events(new_events, record_history=True)
        self._scroll_to_current()

    def _on_consume(self) -> None:
        """Remove the current event then advance to the next one."""
        self._on_del()

    def _on_undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.events.copy())
        prev = self._undo_stack.pop()
        self._set_events(prev, record_history=False)

    def _on_redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.events.copy())
        nxt = self._redo_stack.pop()
        self._set_events(nxt, record_history=False)

    def _on_save(self) -> None:
        self._save_csv()

    def _on_load(self) -> None:
        if self.csv_path is None or not self.csv_path.exists():
            return
        df = pd.read_csv(self.csv_path)
        if not {"start", "stop"}.issubset(df.columns):
            return
        # ``Load`` discards the entire undo/redo history: the on-disk file is
        # treated as the new ground truth.
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._set_events(df, record_history=False)

    # ------------------------------------------------------------------
    # Channel selection
    # ------------------------------------------------------------------
    def refresh_channel_state(self) -> None:
        """Sync the channel list and re-apply the active selection.

        Called by :class:`StimArtefactBinViewer` after the controller's
        ``sort`` runs on each chunk load. Captures the post-sort full trace
        order and re-applies any pending selection so the displayed traces
        stay consistent with the user's choice across reloads.
        """
        if self.ctrl.trace_indices is None:
            return
        self._full_sorted_indices = np.asarray(self.ctrl.trace_indices).copy()
        self._populate_channel_list()
        self._apply_channel_selection()

    @staticmethod
    def _filter_indices(
        full: np.ndarray, selected: set[int] | None
    ) -> np.ndarray:
        """Filter ``full`` (sorted trace indices) by ``selected`` (originals).

        ``None`` selection means show every channel. An empty intersection
        falls back to the full set so the plot is never blank.
        """
        if selected is None:
            return full
        mask = np.isin(full, np.fromiter(selected, dtype=full.dtype))
        if not mask.any():
            return full
        return full[mask]

    def _channel_label(self, trace_idx: int) -> str:
        """Build a short label for ``trace_idx`` from available header keys."""
        header = getattr(self.ctrl.model, "header", None) or {}
        parts: list[str] = [f"#{int(trace_idx)}"]
        for key in ("shank", "x", "y"):
            arr = header.get(key)
            if arr is None:
                continue
            try:
                value = arr[int(trace_idx)]
            except (IndexError, TypeError):
                continue
            if isinstance(value, (int, np.integer)):
                parts.append(f"{key}={int(value)}")
            else:
                try:
                    parts.append(f"{key}={float(value):.0f}")
                except (TypeError, ValueError):
                    parts.append(f"{key}={value}")
        return " ".join(parts)

    def _populate_channel_list(self) -> None:
        """Refresh the QListWidget items from the current sorted trace order."""
        if self._full_sorted_indices is None:
            return
        self._updating_ui = True
        try:
            self.listWidget_stim_channels.clear()
            for trace_idx in self._full_sorted_indices:
                trace_idx = int(trace_idx)
                item = QtWidgets.QListWidgetItem(self._channel_label(trace_idx))
                item.setData(QtCore.Qt.UserRole, trace_idx)
                self.listWidget_stim_channels.addItem(item)
                if (
                    self._selected_traces is None
                    or trace_idx in self._selected_traces
                ):
                    item.setSelected(True)
        finally:
            self._updating_ui = False
        self._update_channel_count_label()
        # Now that real channel labels exist, tighten the group's max width
        # to the widest label rather than the placeholder default.
        new_max = self._compute_channels_max_width()
        self._channels_expanded_max_width = new_max
        if self.groupBox_channels.isChecked():
            self.groupBox_channels.setMaximumWidth(new_max)

    def _compute_channels_max_width(self) -> int:
        """Width sized to the widest item label + scrollbar / margins.

        Falls back to a sensible default when the list is empty (e.g. at
        construction time before any data has been loaded).
        """
        fm = self.listWidget_stim_channels.fontMetrics()
        widest_text = 0
        for i in range(self.listWidget_stim_channels.count()):
            text = self.listWidget_stim_channels.item(i).text()
            widest_text = max(widest_text, fm.horizontalAdvance(text))
        if widest_text == 0:
            widest_text = fm.horizontalAdvance("#999 shank=9 x=999 y=999")
        # Account for the list's frame, scrollbar, and the surrounding
        # group-box margins / title checkbox.
        scrollbar = QtWidgets.QApplication.style().pixelMetric(
            QtWidgets.QStyle.PM_ScrollBarExtent
        )
        return widest_text + scrollbar + 40

    def _update_channel_count_label(self) -> None:
        total = self.listWidget_stim_channels.count()
        if self._selected_traces is None:
            chosen = total
        else:
            chosen = len(self._selected_traces)
        self.label_stim_channel_count.setText(f"{chosen} / {total} channels")

    def _apply_channel_selection(self) -> None:
        """Push the active selection into the controller(s) and redraw.

        Both the image and wiggle controllers carry their own
        ``trace_indices``, so we update both to keep the filter consistent
        across display-mode toggles. ``ControllerWiggle`` has no ``redraw``
        method, so we fall back to ``_update_plotItem``/``set_header`` for
        the wiggle path.
        """
        if self._full_sorted_indices is None:
            return
        idx = self._filter_indices(self._full_sorted_indices, self._selected_traces)
        # Apply to both controllers so toggling display mode keeps the filter.
        for ctrl in (
            getattr(self, "_ctrl_image", None),
            getattr(self, "_ctrl_wiggle", None),
        ):
            if ctrl is not None:
                ctrl.trace_indices = idx
        active = self.ctrl
        if hasattr(active, "redraw"):
            active.redraw()
            n = int(idx.size)
            ymin, ymax = -0.5, max(0.5, n - 0.5)
            self.plotItem_seismic.setLimits(yMin=ymin, yMax=ymax)
            self.plotItem_header_v.setLimits(yMin=ymin, yMax=ymax)
            self.viewBox_seismic.setYRange(ymin, ymax, padding=0)
        else:
            # Wiggle path: re-render trace lines / header strip. Let
            # ``_update_plotItem`` own the y-range so auto-spacing keeps
            # working (it computes spacing from per-trace amplitude and
            # sets its own ``setYRange``); applying image-style integer
            # limits here would clamp the wiggle baselines off-screen.
            # Clear any leftover y-limits from a previous image-mode pass
            # so the auto-space ``setYRange`` isn't clamped to ``[-0.5,
            # ntr - 0.5]``. ``setLimits(yMin=None)`` is a no-op in
            # pyqtgraph; use ``±inf`` to actually disable the clamp.
            self.plotItem_seismic.setLimits(
                yMin=-float("inf"), yMax=float("inf")
            )
            self.plotItem_header_v.setLimits(
                yMin=-float("inf"), yMax=float("inf")
            )
            active._update_plotItem()
            active.set_header()
            # Lock the wiggle view to the actual trace bounds + a small
            # padding so the user can't pan into empty space above/below.
            # Also force a full-zoom-out on every apply so switching INTO
            # wiggle mode (or toggling auto-space) always shows all
            # traces, top-to-bottom, instead of inheriting whatever zoom
            # level was active in density mode.
            n = int(idx.size)
            n_drawn = max(1, n)
            autospace = bool(
                getattr(self, "checkBox_wiggle_autospace", None)
                and self.checkBox_wiggle_autospace.isChecked()
            )
            if autospace:
                # ``ControllerWiggle._update_plotItem`` already called
                # ``setYRange`` based on the auto-spacing it computed,
                # which spans every trace; just read it back.
                ymin, ymax = self.viewBox_seismic.viewRange()[1]
                pad = max((ymax - ymin) * 0.05, 1e-6)
            else:
                # Without auto-space, baselines sit at integer ``y``
                # positions ``0..n_drawn - 1``. Use a half-trace pad so
                # even the top and bottom wiggles have visual breathing
                # room.
                ymin, ymax = -0.5, max(0.5, n_drawn - 0.5)
                pad = max(0.5, (ymax - ymin) * 0.10)
            ylo, yhi = ymin - pad, ymax + pad
            # IMPORTANT: limits BEFORE range. pyqtgraph's ``setLimits``
            # re-clamps the existing range to the new bounds while
            # preserving its span, which silently shifts the view if it
            # runs after ``setYRange``. Locking limits first means the
            # subsequent range call lands at the exact requested values.
            # Also: the ``setYRange`` MUST equal the limits so the view
            # is fully zoomed out (not just "covers" the bounds).
            self.plotItem_seismic.setLimits(yMin=ylo, yMax=yhi)
            self.plotItem_header_v.setLimits(yMin=ylo, yMax=yhi)
            self.viewBox_seismic.setYRange(ylo, yhi, padding=0)
            # Same for x: snap to the full data window so "switch to
            # wiggle" never leaves the user staring at a sliver.
            ns, si, t0 = self.model.ns, self.model.si, self.model.t0
            xmin, xmax = float(t0), float(t0 + ns * si)
            self.viewBox_seismic.setXRange(xmin, xmax, padding=0)

    def _on_channels_all(self) -> None:
        self.listWidget_stim_channels.selectAll()

    def _on_channels_none(self) -> None:
        self.listWidget_stim_channels.clearSelection()

    def _on_channels_apply(self) -> None:
        items = self.listWidget_stim_channels.selectedItems()
        total = self.listWidget_stim_channels.count()
        if not items or len(items) == total:
            self._selected_traces = None
        else:
            self._selected_traces = {
                int(it.data(QtCore.Qt.UserRole)) for it in items
            }
        self._update_channel_count_label()
        self._apply_channel_selection()

    def _on_channels_collapsed_toggled(self, checked: bool) -> None:
        """Show/hide channel content and shrink the group box when collapsed."""
        self._channels_content.setVisible(bool(checked))
        if checked:
            self.groupBox_channels.setMaximumWidth(self._channels_expanded_max_width)
            self.groupBox_channels.setMinimumWidth(0)
        else:
            # Collapse to a thin sliver showing just the title-checkbox.
            self.groupBox_channels.setMaximumWidth(self._channels_collapsed_max_width)
            self.groupBox_channels.setMinimumWidth(self._channels_collapsed_max_width)

    # ------------------------------------------------------------------
    # Overlay (yellow ``stim_artefact_removed`` over wiggle traces)
    # ------------------------------------------------------------------
    def set_overlay_data(self, data: np.ndarray | None) -> None:
        """Store overlay samples (same shape/orientation as ``model.data``).

        Pass ``None`` to clear the overlay. The plot is redrawn so the
        change is visible immediately.
        """
        if data is None:
            self._overlay_data = None
        else:
            arr = np.asarray(data)
            if arr.shape != self.model.data.shape:
                self._overlay_data = None
            else:
                self._overlay_data = arr
        self._refresh_overlay()

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._overlay_enabled = bool(checked)
        if self._overlay_enabled and self._bin_viewer is not None:
            # Ask the bin viewer to load the overlay recording for the
            # current chunk. Failures (missing key, etc.) silently leave
            # the overlay empty.
            try:
                self._bin_viewer.load_overlay_for_current_chunk()
            except Exception:
                self._overlay_data = None
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        """Re-render the active wiggle so the overlay reflects current state."""
        ctrl = getattr(self, "_ctrl_wiggle", None)
        if ctrl is None or self._display_mode != DISPLAY_MODE_WIGGLE:
            # In density mode the overlay is hidden; just clear the line.
            self.plotDataItem_overlay.setVisible(False)
            self.plotDataItem_overlay.clear()
            return
        ctrl._update_plotItem()

    def set_display_mode(self, mode: str) -> None:
        """Toggle wiggle/density and re-apply the active channel filter.

        ``EasyQC.set_display_mode`` calls ``ctrl.set_model(reset_viewbox=False)``
        on the new active controller, which resets its ``trace_indices`` to
        ``arange(ntr)``. We snapshot the post-sort full order and re-apply
        whatever subset the user had selected so switching modes preserves
        their filter.
        """
        super().set_display_mode(mode)
        # Auto-space only makes sense for the wiggle view.
        self._sync_wiggle_widgets_enabled()
        # ``set_model`` re-ran ``editSort`` -> ``trace_indices`` is now the
        # full sorted order on the new controller. Refresh our snapshot
        # without dropping ``_selected_traces``, then re-apply.
        if self.ctrl.trace_indices is not None:
            self._full_sorted_indices = np.asarray(self.ctrl.trace_indices).copy()
            self._populate_channel_list()
            self._apply_channel_selection()

    def _sync_wiggle_widgets_enabled(self) -> None:
        """Enable auto-space / overlay only while wiggle mode is active."""
        is_wiggle = getattr(self, "_display_mode", None) == DISPLAY_MODE_WIGGLE
        for name in (
            "checkBox_wiggle_autospace",
            "checkBox_stim_overlay",
        ):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(is_wiggle)

    def _on_wiggle_layout_changed(self) -> None:
        """Re-apply selection AND force a deterministic full y-rezoom.

        ``_apply_channel_selection`` runs first and clamps the viewbox's
        y-limits to whatever range fit the PREVIOUS autospace
        configuration. Compute the new y-range directly from the freshly
        plotted curve so it always reflects the current settings.
        """
        self._apply_channel_selection()
        if getattr(self, "_display_mode", None) != DISPLAY_MODE_WIGGLE:
            return
        item = getattr(self, "plotDataItem_wiggle", None)
        if item is None:
            return
        y = item.yData
        if y is None or len(y) == 0:
            return
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return
        ylo = float(finite.min())
        yhi = float(finite.max())
        pad = max((yhi - ylo) * 0.05, 0.5)
        ylo -= pad
        yhi += pad
        # Lift any existing clamp so the new range can widen the view,
        # then apply both the view range and the new clamp.
        inf = float("inf")
        self.plotItem_seismic.setLimits(yMin=-inf, yMax=inf)
        self.plotItem_header_v.setLimits(yMin=-inf, yMax=inf)
        self.viewBox_seismic.setYRange(ylo, yhi, padding=0)
        self.plotItem_seismic.setLimits(yMin=ylo, yMax=yhi)
        self.plotItem_header_v.setLimits(yMin=ylo, yMax=yhi)


class _StimControllerWiggle(ControllerWiggle):
    """Wiggle controller that draws an optional yellow overlay.

    Re-uses ``ControllerWiggle._update_plotItem`` for the main traces, then
    paints ``view._overlay_data`` (e.g. ``stim_artefact_removed``) on top
    using the same baseline / spacing / gain so the two signals align.
    """

    def _update_plotItem(self, tlim=None, clim=None):
        super()._update_plotItem(tlim=tlim, clim=clim)
        view = self.view
        item = getattr(view, "plotDataItem_overlay", None)
        if item is None:
            return
        if (
            not getattr(view, "_overlay_enabled", False)
            or view._overlay_data is None
            or self.model.taxis != 0
            or view._overlay_data.shape != self.model.data.shape
        ):
            item.setVisible(False)
            item.clear()
            return

        # Mirror the baseline / spacing computation in
        # ``ControllerWiggle._update_plotItem`` so the overlay sits on
        # the same baselines as the raw traces.
        idx = (
            self.trace_indices
            if self.trace_indices is not None
            else np.arange(self.model.ntr)
        )
        raw = self.model.data[:, idx]
        ov = view._overlay_data[:, idx]
        ntr = ov.shape[1]
        if ntr == 0 or ov.size == 0:
            item.setVisible(False)
            item.clear()
            return

        gain_div = 10 ** (self.gain / 20)
        autospace = bool(
            getattr(self.view, "checkBox_wiggle_autospace", None)
            and self.view.checkBox_wiggle_autospace.isChecked()
        )
        if autospace:
            ptp = np.nanmax(raw, axis=0) - np.nanmin(raw, axis=0)
            spacing = float(np.nanmax(ptp) / gain_div) if ptp.size else 1.0
            if not np.isfinite(spacing) or spacing <= 0:
                spacing = 1.0
        else:
            spacing = 1.0
        # Centre the overlay on each trace's slot using its own DC, so
        # any residual offset between raw and overlay is preserved.
        ov = ov - np.nanmean(ov, axis=0, keepdims=True)
        if autospace:
            # Use the raw amplitude for normalisation so traces are
            # scaled identically to the main wiggle (overlay rides at
            # the same visual amplitude).
            ptp_per = np.nanmax(raw, axis=0) - np.nanmin(raw, axis=0)
            ptp_per = np.where(ptp_per > 0, ptp_per, 1.0)
            ov = ov / ptp_per * (spacing * 0.9 * gain_div)
        wiggle_y = np.r_[ov, np.full((1, ntr), np.nan)]
        wiggle_y = (
            wiggle_y / gain_div + (np.arange(ntr) * spacing)[np.newaxis, :]
        )
        item.setData(
            x=np.tile(np.r_[self.tscale, np.nan], ntr),
            y=wiggle_y.T.flatten(),
        )
        item.setVisible(True)


class StimArtefactBinViewer(EphysBinViewer):
    """SpikeInterface-backed window for stim-artefact inspection.

    Holds the binary navigation slider; the actual region editing lives on the
    spawned :class:`StimArtefactViewer`. The events table and CSV path are
    handed off to the viewer once on first display.
    """

    REQUIRED_RECORDINGS = ("raw", "stim_artefact_removed")

    def __init__(
        self,
        recordings_dict: dict,
        filepath: str | Path,
        viewer_key: str = "raw",
        *args,
        **kwargs,
    ) -> None:
        self._validate_recordings_dict(recordings_dict)
        if viewer_key not in recordings_dict:
            raise ValueError(
                f"viewer_key must be one of {tuple(recordings_dict.keys())}, got {viewer_key!r}."
            )

        self.csv_path = Path(filepath)
        self.events = pd.read_csv(self.csv_path)
        if not {"start", "stop"}.issubset(self.events.columns):
            raise ValueError(
                f"{self.csv_path} must contain 'start' and 'stop' columns; got "
                f"{tuple(self.events.columns)}."
            )
        self.events = (
            self.events.loc[:, ["start", "stop"]]
            .dropna()
            .sort_values("start", kind="mergesort")
            .reset_index(drop=True)
        )
        self.viewer_key = viewer_key
        super().__init__(None, *args, **kwargs)
        self.settings = QtCore.QSettings("int-brain-lab", "StimArtefactBinViewer")
        self.setWindowTitle("Stim Artefact Viewer")
        self.actionopen.setEnabled(False)
        self.actionopen_live_recording.setEnabled(False)
        self.data = SpikeInterfaceDataModel(recordings_dict)
        self._setup_viewers_and_checkboxes()
        self._setup_slider()

    @classmethod
    def _validate_recordings_dict(cls, recordings_dict: dict) -> None:
        missing = [key for key in cls.REQUIRED_RECORDINGS if key not in recordings_dict]
        if missing:
            raise ValueError(
                "recordings_dict must contain the required keys "
                f"{cls.REQUIRED_RECORDINGS}; missing {tuple(missing)}."
            )

    def _setup_viewers_and_checkboxes(self) -> None:
        self.frame.hide()
        self.viewers = {self.viewer_key: None}
        self.cbs = {}

    def set_viewer_key(self, viewer_key: str) -> None:
        if viewer_key not in self.data.recordings_dict:
            raise ValueError(
                f"viewer_key must be one of {tuple(self.data.recordings_dict.keys())}, "
                f"got {viewer_key!r}."
            )
        self.viewer_key = viewer_key

    def jump_to_time(self, t: float) -> None:
        """Center the loaded window on the requested absolute time."""
        sampling_frequency = self.data.get_sampling_frequency()
        num_samples = self.data.get_num_samples()
        requested_sample = int(round(float(t) * sampling_frequency))
        requested_sample = max(0, min(requested_sample, int(num_samples) - 1))
        max_first = max(0, int(num_samples) - NSAMP_CHUNK)
        first_sample = requested_sample - NSAMP_CHUNK // 2
        first_sample = max(0, min(first_sample, max_first))
        center_time = requested_sample / sampling_frequency
        self._first_sample = first_sample
        slider_value = int(round(first_sample / NSAMP_CHUNK))
        slider_value = max(0, min(slider_value, self.horizontalSlider.maximum()))
        self.horizontalSlider.blockSignals(True)
        self.horizontalSlider.setValue(slider_value)
        self.horizontalSlider.blockSignals(False)
        self._update_time_label()
        self.on_horizontalSliderReleased(center_time=center_time)

    OVERLAY_KEY = "stim_artefact_removed"

    def load_overlay_for_current_chunk(self) -> None:
        """Fetch the overlay recording for the active chunk and push it.

        No-op if the active viewer key is itself the overlay key (the
        overlay would duplicate the main signal) or if the overlay key
        is missing from ``recordings_dict``.
        """
        viewer = self.viewers.get(self.viewer_key)
        if viewer is None:
            return
        if (
            self.viewer_key == self.OVERLAY_KEY
            or self.OVERLAY_KEY not in self.data.recordings_dict
        ):
            viewer.set_overlay_data(None)
            return
        first = int(self._first_sample)
        last = first + int(NSAMP_CHUNK)
        ov = self.data.get_data(first, last, self.OVERLAY_KEY)
        # ``viewseis`` stores ``data.T * a_scalar`` on the model, so apply
        # the same transform here for shape/scale consistency.
        viewer.set_overlay_data(np.asarray(ov).T * A_SCALAR)

    def on_horizontalSliderReleased(
        self, center_time: float | None = None
    ) -> None:
        prev = self.viewers.get(self.viewer_key)
        prev_range = None
        if prev is not None and prev.isVisible():
            xr, yr = prev.viewBox_seismic.viewRange()
            prev_range = (list(xr), list(yr))

        first = int(self._first_sample)
        last = first + int(NSAMP_CHUNK)
        t0 = first / self.data.get_sampling_frequency()
        data = self.data.get_data(first, last, self.viewer_key)

        viewer = _view_stim_artefact(
            data,
            self.data.get_sampling_frequency(),
            channels=self.data.get_header(),
            title=self.viewer_key,
            t0=t0 * T_SCALAR,
            t_scalar=T_SCALAR,
            a_scalar=A_SCALAR,
        )
        viewer.ctrl.sort(["!x", "y", "shank"])
        # Refresh the channel list / re-apply any active selection now that
        # the controller's sorted trace order is settled for this chunk.
        viewer.refresh_channel_state()

        # Hand off the events table on first display, then keep regions in
        # sync with the new chunk on every subsequent reload.
        first_attach = viewer is not prev
        if first_attach or viewer.csv_path is None:
            viewer.attach_events(self.events, self.csv_path, self)
            # Keep our local handle pointing at the viewer's authoritative
            # frame so external callers don't see a stale copy.
            self.events = viewer.events
        else:
            self.events = viewer.events
            viewer.refresh_regions()

        self.viewers[self.viewer_key] = viewer

        # Re-load the yellow overlay (stim_artefact_removed) when the
        # toggle is on so it tracks the new chunk window.
        if getattr(viewer, "_overlay_enabled", False):
            self.load_overlay_for_current_chunk()

        if prev_range is None:
            return

        xr_prev, yr_prev = prev_range
        width = xr_prev[1] - xr_prev[0]
        xmin, xmax = viewer.ctrl.limits()[0]
        if center_time is None:
            new_x0 = t0 * T_SCALAR
        else:
            new_x0 = center_time * T_SCALAR - width / 2

        if width >= xmax - xmin:
            new_x0, new_x1 = xmin, xmax
        else:
            new_x0 = max(xmin, min(new_x0, xmax - width))
            new_x1 = new_x0 + width
        viewer.viewBox_seismic.setXRange(new_x0, new_x1, padding=0)
        viewer.viewBox_seismic.setYRange(yr_prev[0], yr_prev[1], padding=0)
        viewer.refresh_regions()


def _view_stim_artefact(
    data: np.ndarray,
    fs: float,
    channels: dict | None = None,
    br=None,
    title: str = "stim_artefact",
    t0: float = 0.0,
    t_scalar: float = T_SCALAR,
    a_scalar: float = A_SCALAR,
    colormap: str | pg.ColorMap | matplotlib.colors.Colormap | None = None,
) -> StimArtefactViewer:
    """Create a StimArtefactViewer window to display an array of data."""

    create_app()
    ev = StimArtefactViewer._get_or_create(title=title)

    if data is not None:
        ev.model.set_data(data.T * a_scalar, si=1 / fs, header=channels, t0=t0, taxis=0)
        ev.ctrl.set_model()

    ev.show()
    if colormap is not None:
        ev.setColorMap(colormap)

    return ev
