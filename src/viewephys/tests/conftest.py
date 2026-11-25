import pytest
from qtpy import QtCore

_real_qsettings = QtCore.QSettings


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path, monkeypatch):
    """Route all QSettings to a per-test ini file.

    The app constructs ``QSettings(org, app)``, which on macOS always maps to
    the native plist in ~/Library/Preferences and ignores ``setPath``; tests
    would therefore read and overwrite the real user preferences.  Patching
    the constructor keeps every test's settings in a throwaway tmp dir.
    """
    ini = tmp_path / "settings.ini"

    def qsettings_factory(*args, **kwargs):
        return _real_qsettings(str(ini), _real_qsettings.Format.IniFormat)

    monkeypatch.setattr(QtCore, "QSettings", qsettings_factory)
