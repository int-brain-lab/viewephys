from viewephys.gui import EphysBinViewer
from pathlib import Path

example_file_path = Path(__file__) / "example_bin" / "1119617_LSE1_shank12_g0_t0.imec0.ap"

viewer = EphysBinViewer(example_file_path)

