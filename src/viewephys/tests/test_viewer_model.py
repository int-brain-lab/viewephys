import numpy as np

from viewephys.tests.test_viewer_helpers import synthetic_seismic_data
from viewephys.viewer.gui import Model


def test_get_trace():
    ntr = 400
    ns = 2500
    si = 0.002

    data, header = synthetic_seismic_data(ntr=ntr, ns=ns)
    model = Model(data=data, si=si, header=header)
    model.set_data(data=data, si=si, header=header)

    np.testing.assert_array_equal(
        model.get_trace(50.02, neighbors=0).T, model.data[50, :]
    )
    np.testing.assert_array_equal(
        model.get_trace(0.02, neighbors=1).T, model.data[:2, :]
    )
    np.testing.assert_array_equal(
        model.get_trace(40.02, neighbors=1).T, model.data[39:42, :]
    )
