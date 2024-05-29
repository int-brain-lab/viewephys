from viewephys.gui import PickSpikes
import numpy as np

ps = PickSpikes()
DEFAULT_DF_COLUMNS = ['sample', 'trace', 'amp', 'group']

def test_init_df(ps):
    df = ps.init_df(nrow=0)
    # Check size
    np.testing.assert_equal(df.shape[0], 0)
    # Check column names
    indxmissing = np.where(~df.columns.isin(DEFAULT_DF_COLUMNS))[0]
    np.testing.assert_(len(indxmissing) == 0)


def test_new_row_frompick():
    new_row = ps.new_row_frompick(sample=1, trace=2, amp=3, group=4)
    # Check size
    np.testing.assert_(new_row.shape[0] == 1)
    # Check column names
    indxmissing = np.where(~new_row.columns.isin(DEFAULT_DF_COLUMNS))[0]
    np.testing.assert_(len(indxmissing) == 0)
    # Check values
    np.testing.assert_(new_row['sample'][0] == 1)
    np.testing.assert_(new_row['trace'][0] == 2)
    np.testing.assert_(new_row['amp'][0] == 3)
    np.testing.assert_(new_row['group'][0] == 4)



