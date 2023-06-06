"""
Command line interface to the viewephys GUI
# viewephys -f /path/to/file.bin
# viewephys -f /path/to/file.cbin
"""
import argparse
import easyqc.qt
from viewephys.gui import EphysBinViewer
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'


def viewephys():
    """
    This command will open an empty GUI with a menu file that allows to load a flat binary
    file readable by ibllib.io.spikeglx.Reader
    :return:
    """
    parser = argparse.ArgumentParser(description='Electrophysiology file viewer with preprocessing options')
    parser.add_argument('-f', '--file', default=None, help='path to the binary file to load', required=False)
    args = parser.parse_args()  # returns data from the options specified (echo)
    print(args.file)
    app = easyqc.qt.create_app()
    self = EphysBinViewer(args.file)  # noqa
    sys.exit(app.exec())


if __name__ == "__main__":
    viewephys()
