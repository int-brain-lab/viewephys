"""
This example shows how to create an Ephys binary viewer from a script.
This is an alternative to starting the viewer through the command line.
"""

from pathlib import Path

from viewephys.gui import EphysBinViewer, create_app

app = create_app()

viewer = EphysBinViewer(
    Path(__file__).parent / "example_bin" / "1119617_LSE1_shank12_g0_t0.imec0.ap.bin"
)

app.exec()
