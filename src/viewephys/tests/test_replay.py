import sys
import threading
from pathlib import Path

import pytest
from spikeglx import _mock_spikeglx_file

from viewephys.command_line import replay
from viewephys.replay import write_replay

META = Path(__file__).parent / "mock_data" / "meta_file.meta"


@pytest.fixture
def source_bin(tmp_path):
    (tmp_path / "src").mkdir()
    info = _mock_spikeglx_file(tmp_path / "src" / "rec.ap.bin", META, 3000, 385)
    return Path(info["bin_file"])


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["viewephys-replay", *map(str, argv)])
    replay()


def test_replay_writes_to_out_dir(source_bin, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _run(monkeypatch, source_bin, "-o", out, "--speed", 0)

    dest = out / "rec.ap.bin"
    assert dest.read_bytes() == source_bin.read_bytes()
    assert (out / "rec.ap.meta").exists()
    # the path printed for the GUI must be the one actually written
    assert str(dest) in capsys.readouterr().out


def test_replay_default_out_dir(source_bin, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(monkeypatch, source_bin, "--speed", 0)
    assert (tmp_path / "rec_replay" / "rec.ap.bin").exists()


def test_replay_missing_source(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path / "nope.ap.bin", "-o", tmp_path / "out")
    assert exc.value.code == 2
    assert "no such file" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_write_replay_copies_file(source_bin, tmp_path):
    dest = write_replay(source_bin, tmp_path / "out", speed=0, chunk_s=0.01)
    assert dest == tmp_path / "out" / "rec.ap.bin"
    assert dest.read_bytes() == source_bin.read_bytes()
    assert (tmp_path / "out" / "rec.ap.meta").exists()


def test_write_replay_stops_on_event(source_bin, tmp_path):
    stop = threading.Event()
    stop.set()
    dest = write_replay(source_bin, tmp_path / "out", speed=0, stop=stop)
    assert dest.stat().st_size == 0
