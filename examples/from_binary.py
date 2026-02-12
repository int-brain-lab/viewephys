from viewephys.gui import EphysBinViewer, create_app
from pathlib import Path

app = create_app()

viewer = EphysBinViewer(Path(__file__).parent / "example_bin" / "1119617_LSE1_shank12_g0_t0.imec0.ap.bin")

app.exec()