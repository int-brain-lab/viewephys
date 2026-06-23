from contextlib import suppress
from pathlib import Path

from qtpy import QtCore

from viewephys.data_model import SpikeInterfaceDataModel
from viewephys.gui import A_SCALAR, T_SCALAR, EphysBinViewer, viewephys
from viewephys.stim_artefact_viewer import stim_artefact_viewer

# Pre-expected recording names in ``recordings_dict``: "raw" is shown in the
# stim viewer, and "stim_artefact_removed" is overlaid on top of it in yellow.
RAW_KEY = "raw"
STIM_REMOVED_KEY = "stim_artefact_removed"


class StimArtefactBinViewer(EphysBinViewer):
    """"""

    def __init__(
        self,
        recordings_dict: dict,
        filepath: str | Path,
        *args,
        **kwargs,
    ) -> None:
        # FIX NAMING
        self.csv_path = Path(filepath)

        # The dict must at least carry the raw band and the stim-removed band.
        missing = {RAW_KEY, STIM_REMOVED_KEY} - set(recordings_dict)
        if missing:
            raise ValueError(
                "recordings_dict must contain the keys "
                f"{RAW_KEY!r} and {STIM_REMOVED_KEY!r}; missing: {sorted(missing)}"
            )

        super().__init__(recordings_dict, *args, **kwargs)

        # The .ui defines a fixed window height, but the recording-step
        # checkboxes are added dynamically (one per recording) for
        # spikeinterface, so the group box height varies. Snap only the height
        # to the minimum the content needs (keeping the predefined width).
        # Deferred so it runs after the layout has been activated; an immediate
        # call would use the pre-layout geometry and leave slack.
        QtCore.QTimer.singleShot(0, self._snap_height_to_content)

    def _snap_height_to_content(self) -> None:
        self.layout().activate()
        self.resize(self.width(), self.minimumSizeHint().height())

    def on_stim_viewer_jump_requested(self, t: float) -> None:
        # ``t`` is the region centre; the jump time is the window first sample,
        # so offset by half the window to keep the region centred in the view.
        window_seconds = self.window_length_n / self.data.get_sampling_frequency()
        first_sample_time = t - window_seconds / 2
        self.lineEdit_jumpTime.setText(f"{first_sample_time:.6f}")
        self.on_jumpToTimeRequested()

    # TODO: we can do some refactoring here!
    def on_horizontalSliderReleased(  # noqa: C901
        self, center_time: float | None = None, reset_zoom: bool = False
    ) -> None:
        """
        Open EphysViewer windows at the selected timepoint
        for the selected preprocessing steps.

        The horizontal slider opens EphysViewer windows starting at the
        selected timepoint. Jump requests may pass ``center_time`` so existing
        viewer zoom is preserved around the requested absolute time. When
        ``reset_zoom`` is True (e.g. after a window-length change) the previous
        zoom is discarded so the freshly loaded window is shown in full.

        Depending on the selected preprocessing steps, open a number of
        viewers (one for each selected preprocessing step) to display the
        preprocessed data at the selected timepoint.
        """
        # Capture current zoom per visible viewer so we can restore it after the reload.
        prev_ranges: dict[str, tuple[list[float], list[float]] | None] = {}
        for k, ev in self.viewers.items():
            if ev is not None and ev.isVisible():
                xr, yr = ev.viewBox_seismic.viewRange()
                prev_ranges[k] = (list(xr), list(yr))
            else:
                prev_ranges[k] = None

        first = int(self._first_sample)
        last = first + int(self.window_length_n)
        t0 = float(self.data.get_times()[first])
        # Timestamp of sample 0: the events file stores absolute sample indices,
        # which must be offset by this when converted to seconds.
        first_time = float(self.data.get_times()[0])

        # Old data flow preserved: fetch raw once, branch per preprocessing step.
        # A fully general interface would re-fetch inside each get_data() call
        # (matching SI semantics) but the performance impact is small for short
        # snippets and the shared raw buffer avoids redundant disk reads.
        raw = self.data.get_raw(first, last)

        # get parameters for both AP and LFP band

        # For each preprocessing step, create an EphysViewer
        for k in self.viewers:
            if not self.cbs[k].isChecked():
                continue

            data = self.data.get_data(first, last, k, raw=raw)

            if k == "raw":
                # Source the overlay from the pre-expected stim-removed band in
                # the recordings dict, over the same window as the raw data.
                data_stim_removed = self.data.get_data(
                    first, last, STIM_REMOVED_KEY, raw=raw
                )
                viewer = stim_artefact_viewer(
                    data,
                    self.data.get_sampling_frequency(),
                    channels=self.data.get_header(),
                    events_path=self.csv_path,
                    title=k,
                    t0=t0 * T_SCALAR,
                    first_time=first_time,
                    t_scalar=T_SCALAR,
                    a_scalar=A_SCALAR,
                    data_stim_removed=data_stim_removed,
                )
                with suppress(TypeError):
                    viewer.request_jump_to_time.disconnect(
                        self.on_stim_viewer_jump_requested
                    )
                viewer.request_jump_to_time.connect(self.on_stim_viewer_jump_requested)

            else:
                viewer = viewephys(
                    data,
                    self.data.get_sampling_frequency(),
                    channels=self.data.get_header(),
                    title=k,
                    t0=t0 * T_SCALAR,
                    t_scalar=T_SCALAR,
                    a_scalar=A_SCALAR,
                )
            if isinstance(self.data, SpikeInterfaceDataModel):
                # reorder the spikeinterface channel ordering to match SpikeGLXReader
                viewer.ctrl.sort(["!x", "y", "shank"])

            self.viewers[k] = viewer

            prev = prev_ranges.get(k)
            if not reset_zoom and prev is not None:
                xr_prev, yr_prev = prev
                width = xr_prev[1] - xr_prev[0]
                xmin, xmax = viewer.ctrl.limits()[0]
                if center_time is None:
                    new_x0 = t0 * T_SCALAR
                else:
                    new_x0 = center_time * T_SCALAR - width / 2

                # Preserve the previous zoom width without panning past the data.
                if width >= xmax - xmin:
                    new_x0, new_x1 = xmin, xmax
                else:
                    new_x0 = max(xmin, min(new_x0, xmax - width))
                    new_x1 = new_x0 + width
                viewer.viewBox_seismic.setXRange(new_x0, new_x1, padding=0)
                viewer.viewBox_seismic.setYRange(yr_prev[0], yr_prev[1], padding=0)
