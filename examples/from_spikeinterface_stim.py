"""
Small example showing how to load a SpikeGLX recording through
SpikeInterface, preprocess it, and open the stim artefact viewer.
"""

from pathlib import Path

import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro

import viewephys.gui as _vgui
import viewephys.stim_artefact as _vstim

from viewephys.stim_artefact import StimArtefactBinViewer
from viewephys.viewer.qt import create_app

# We must create the Qt application before showing the viewer window.
app = create_app()

raw_data_path = Path(__file__).parent / "example_bin"
csv_path = Path(__file__).parent / "example_stim_events.csv"

# Load the raw AP stream through SpikeInterface.
rec_raw = si_extractors.read_spikeglx(raw_data_path, stream_id="imec0.ap")

# Show ~0.5 s per chunk regardless of sample rate. NSAMP_CHUNK is a module
# constant in viewephys.gui that's also re-imported into stim_artefact, so we
# patch both module bindings before any viewer is constructed.
_chunk = int(round(0.5 * rec_raw.get_sampling_frequency()))
_vgui.NSAMP_CHUNK = _chunk
_vstim.NSAMP_CHUNK = _chunk

# Build a simple cleaned recording to use as the stim-artefact-removed layer.
rec_shift = si_prepro.phase_shift(rec_raw)
rec_filt = si_prepro.bandpass_filter(rec_shift, freq_min=300, freq_max=6000)
rec_clean = si_prepro.common_reference(rec_filt, operator="median")

viewer = StimArtefactBinViewer(
    {
        "raw": rec_raw,
        "stim_artefact_removed": rec_clean,
    },
    filepath=csv_path,
)

app.exec()
