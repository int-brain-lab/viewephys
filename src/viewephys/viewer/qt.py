import logging
from functools import wraps

from qtpy import QtWidgets

_logger = logging.getLogger("ibllib")


def get_main_window():
    """Get the main window of a Qt application."""
    app = QtWidgets.QApplication.instance()
    return [w for w in app.topLevelWidgets() if isinstance(w, QtWidgets.QMainWindow)][0]


def create_app():
    """Create a Qt application."""
    global QT_APP
    QT_APP = QtWidgets.QApplication.instance()
    if QT_APP is None:  # pragma: no cover
        QT_APP = QtWidgets.QApplication([])
    return QT_APP


def require_qt(func):
    """Ensure a Qt application exists before calling the wrapped function."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        if not QtWidgets.QApplication.instance():  # pragma: no cover
            _logger.warning("Creating a Qt application.")
            create_app()
        return func(*args, **kwargs)

    return wrapped


@require_qt
def run_app():  # pragma: no cover
    """Run the Qt application."""
    global QT_APP
    return QT_APP.exit(QT_APP.exec_())
