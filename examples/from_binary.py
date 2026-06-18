"""
This example shows how to create an Ephys binary viewer from a script.
This is an alternative to starting the viewer through the command line.
"""

from viewephys.gui import EphysBinViewer, create_app

app = create_app()

viewer = EphysBinViewer(
    r"C:\Users\Joe\Desktop\ses-03_g0\ses-03_g0_imec0\ses-03_g0_t0.imec0.ap.bin"
    # Path(__file__).parent / "example_bin" / "1119617_LSE1_shank12_g0_t0.imec0.ap.bin"
)

app.exec()
