# Changelog

## [Unreleased]

### fixed
- Pick-spikes auto-detect window now scales with the current zoom level instead of using a fixed half-second range, preventing false `out_of_time_range` rejections at high sample rates (thanks @JoeZiminski)
- `QSettings.value()` cast to `int` for `nfft`/`nperseg` in `ImShowSpectrogram` to avoid `TypeError` on `//` operator

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
