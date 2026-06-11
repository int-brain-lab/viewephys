"""
Small example showing how to load a SpikeGLX recording through
SpikeInterface, preprocess it, and open the stim artefact viewer.
"""

from pathlib import Path

import spikeinterface as si
import spikeinterface.preprocessing as si_prepro

from viewephys.stim_artefact_bin import StimArtefactBinViewer
from viewephys.viewer.qt import create_app

# We must create the Qt application before showing the viewer window.
app = create_app()

csv_path = Path(__file__).parent / "example_stim_events.csv"

# Build a synthetic recording long enough to inspect events after 10 seconds.
rec_raw, _sorting = si.generate_ground_truth_recording(
    num_channels=64,
    sampling_frequency=30_000.0,
    durations=[12.0],
    num_units=10,
    seed=0,
)

# Build a simple cleaned recording to use as the stim-artefact-removed layer.
rec_filt = si_prepro.bandpass_filter(rec_raw, freq_min=300, freq_max=6000)
rec_clean = si_prepro.common_reference(rec_filt, operator="median")

viewer = StimArtefactBinViewer(
    {
        "raw": rec_raw,
        "stim_artefact_removed": rec_clean,
    },
    filepath=csv_path,
)

app.exec()
