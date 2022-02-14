import easyqc.qt
import numpy as np
import viewephys.gui as gui
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'


def viewephys():
    """
    This command will open an empty GUI with a menu file that allows to load a flat binary
    file readable by ibllib.io.spikeglx.Reader
    :return:
    """
    # QApplication.setStyle('Fusion')
    ev = gui.viewephys(data=np.random.randn(20, 10), fs=30000)
    # the quit function will stop the app if it is provided in the QT_APP property
    ev.QT_APP = easyqc.qt.create_app()
    ev.show()
    sys.exit(ev.QT_APP.exec())
