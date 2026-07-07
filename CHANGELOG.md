# Changelog

## [Unreleased]

## [1.3.0] - 2026-07-07

### added
- `examples/` folder with `from_binary.py` and `from_array.py` demonstrating script usage ([#29](https://github.com/int-brain-lab/viewephys/pull/29), thanks @JoeZiminski)
- README expanded with script-usage instructions showing the `create_app()` / `app.exec()` pattern
- `spikeinterface` added as a dependency
- `SpikeGLXDataModel` interface layer decouples data access from the viewer, laying the groundwork for a future SpikeInterface backend ([#30](https://github.com/int-brain-lab/viewephys/pull/30), thanks @JoeZiminski)
- Channels can now be sorted in descending order by prefixing the sort key with `!` (e.g. `"!depth"`); `lexsort` only supports ascending order, so the key values are negated before sorting ([#32](https://github.com/int-brain-lab/viewephys/pull/32), thanks @JoeZiminski)

### changed
- EphysBinViewer main window reorganised: the separate "Jump to" field has been merged into the slider value lineedit — the current sample is shown there directly and typing a time then pressing Enter navigates to that position ([#38](https://github.com/int-brain-lab/viewephys/pull/38), thanks @JoeZiminski)

### fixed
- Pick-spikes auto-detect window now scales with the current zoom level instead of using a fixed half-second range, preventing false `out_of_time_range` rejections at high sample rates ([#28](https://github.com/int-brain-lab/viewephys/pull/28), thanks @JoeZiminski)
- `QSettings.value()` cast to `int` for `nfft`/`nperseg` in `ImShowSpectrogram` to avoid `TypeError` on `//` operator ([#28](https://github.com/int-brain-lab/viewephys/pull/28))
- Duplicate `groupBox` widget name in `nav_file.ui` that caused a Qt warning on startup
- Header plots are now flush with the seismic display: the vertical header's bottom axis is given an empty label so it reserves the same height as the seismic's "Time (s)" label, keeping the two ViewBoxes vertically aligned ([#43](https://github.com/int-brain-lab/viewephys/issues/43))

## [1.2.0] - 2026-04-30

### added
- "Jump to" time entry in the EphysBinViewer for navigating to a specific timepoint
- Jump-to navigation loads the window starting at the closest sample, rather than snapping to the horizontal slider's 10000-sample discretisation
- Preserve the current zoom range across slider/jump-to reloads in the EphysBinViewer

## [1.1.2] - 2026-04-21

### changed
- Removed the easyqc dependency; tests previously sourced from easyqc are now imported directly

### added
- Automatic time axis label in viewer

### fixed
- Scrollbar behaviour
- Visibility of viewer menus

## [1.1.1] 2026-02-16

### added
- option for unfiltered data and LFP filtered data in file viewer
- Wiggle display option via EasyQC 1.3.0

## [1.1.0 YANKED pdm distribution]  2026-02-16

## [1.0.1] - 2024-12-06

### added
- minimal support for Open Ephys [issue 7](https://github.com/int-brain-lab/viewephys/issues/7)

## [1.0.0] - 2024-10-09

### added
- add spike groups in the picking interface
- add the live recording option in the EphysBinViewer
-
