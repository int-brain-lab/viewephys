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
    app = easyqc.qt.create_app()
    self = EphysBinViewer()
    # the quit function will stop the app if it is provided in the QT_APP property
    sys.exit(app.exec())
