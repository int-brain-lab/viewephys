"""
Command line interface to the viewephys GUI
# viewephys -f /path/to/file.bin
# viewephys -f /path/to/file.cbin
"""

import argparse
import os
import sys

os.environ["QT_MAC_WANTS_LAYER"] = "1"


def viewephys():
    """
    This command will open an empty GUI with a menu file that allows to
    load a flat binary file readable by ibllib.io.spikeglx.Reader
    :return:
    """
    parser = argparse.ArgumentParser(
        description="Electrophysiology file viewer with preprocessing options"
    )
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help="path to the binary or lfpack .h5 file to load",
        required=False,
    )
    parser.add_argument(
        "--lfpack",
        action="store_true",
        help="open the lfpack HDF5 viewer (auto-selected for .h5/.hdf5 files)",
    )
    args = parser.parse_args()  # returns data from the options specified (echo)
    print(args.file)
    from pathlib import Path

    from viewephys.gui import EphysBinViewer, LFPackBinViewer
    from viewephys.viewer.qt import create_app

    app = create_app()
    # lfpack HDF5 archives need the LFPack-aware viewer; binaries use the base one.
    is_h5 = args.file is not None and Path(args.file).suffix.lower() in (
        ".h5",
        ".hdf5",
    )
    ViewerClass = LFPackBinViewer if (args.lfpack or is_h5) else EphysBinViewer
    self = ViewerClass(args.file)  # noqa
    sys.exit(app.exec())


if __name__ == "__main__":
    viewephys()
