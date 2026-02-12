import numpy as np
import pandas as pd
from brainbox.tests.test_metrics import multiple_spike_trains
from iblutil.util import Bunch
from neuropixel import trace_header

from viewephys.raster import ProbeData


def test_model_dataclass():
    st, sa, sc = multiple_spike_trains()
    spikes = Bunch(dict(times=st, clusters=sc, amps=sa, depths=sa * 0 + 100))
    clusters = Bunch(dict(channels=np.random.randint(0, 384, np.max(sc))))
    channels = Bunch(trace_header(version=1))

    ProbeData(spikes=spikes, clusters=clusters, channels=channels)
    ProbeData(
        spikes=pd.DataFrame(spikes),
        clusters=pd.DataFrame(clusters),
        channels=pd.DataFrame(channels),
    )
