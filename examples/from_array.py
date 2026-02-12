from viewephys.gui import viewephys, create_app
from pathlib import Path
import spikeinterface.extractors as si_extractors
import spikeinterface.preprocessing as si_prepro

# we must create the app before creating the viewephys windows
app = create_app()

# Load and preprocess data using spikeinterface
raw_data_path = Path(__file__).parent / "example_bin"

rec_raw = si_extractors.read_spikeglx(raw_data_path, stream_id="imec0.ap")
fs = rec_raw.get_sampling_frequency()

rec_shift = si_prepro.phase_shift(rec_raw)
rec_filt = si_prepro.bandpass_filter(rec_shift, freq_min=300, freq_max=6000)
rec_cmr = si_prepro.common_reference(rec_filt, operator="median")

# Note the transpose, spikeinterface returns (time, channel)
# but we need (channel, time)
filt_data = rec_filt.get_traces().T
cmr_data = rec_cmr.get_traces().T

# View the data at difference stages of preprocessing
view_shift = viewephys(filt_data, fs=fs, title="view filt")
view_filt = viewephys(cmr_data, fs=fs, title="view CMR")

app.exec()