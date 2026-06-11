"""
This example shows how to create an Ephys binary viewer from a script.
This is an alternative to starting the viewer through the command line.
"""

from pathlib import Path

from viewephys.gui import EphysBinViewer, create_app
from viewephys.stim_artefact import StimArtefactBinViewer
import spikeinterface as si
import spikeinterface.extractors as se


app = create_app()

raw_rec = se.read_openephys(r"D:\EPhys\raw\EC19\2025-10-04_15-05-38\Record Node 102\experiment1\recording1", stream_name="Record Node 102#Acquisition_Board-100.acquisition_board_ADC")
print("has probe: ", raw_rec.has_probe())

rec_hp = si.load(r"D:\EPhys\raw\EC19\2025-10-04_15-05-38\hp_filt")

viewer = StimArtefactBinViewer(
    {
        "raw": rec_hp,
        "stim_artefact_removed": rec_hp,
    },
    filepath=r"D:\EPhys\raw\EC19\2025-10-04_15-05-38\artifacts_split1.csv",
)
app.exec()

