"""Create an Ephys viewer from SpikeInterface recordings."""

import spikeinterface as si
import spikeinterface.preprocessing as spre

from viewephys.gui import EphysBinViewer, create_app

app = create_app()

raw, _sorting = si.generate_ground_truth_recording(
    num_channels=383,
    sampling_frequency=30_000.0,
    durations=[25.0],
    num_units=10,
    seed=0,
)

# Set times so they don't start at zero
raw.set_times(raw.get_times() + 40)

highpass = spre.highpass_filter(raw, freq_min=300.0)

viewer = EphysBinViewer(
    {
        "raw": raw,
        "highpass": highpass,
    }
)

app.exec()
