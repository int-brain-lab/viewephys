import threading
from pathlib import Path

import pytest
from spikeglx import _mock_spikeglx_file

from viewephys.replay import write_replay

META = Path(__file__).parent / "mock_data" / "meta_file.meta"


@pytest.fixture
def source_bin(tmp_path):
    (tmp_path / "src").mkdir()
    info = _mock_spikeglx_file(tmp_path / "src" / "rec.ap.bin", META, 3000, 385)
    return Path(info["bin_file"])


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
