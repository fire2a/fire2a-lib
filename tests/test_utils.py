#!python3

from fire2a.utils import loadtxt_nodata

def test_loadtxt_nodata():
    from numpy.random import randint, random
    from numpy import int32, all
    from io import StringIO
    # Test loadtxt_safer
    data = randint(0, 10, (3, 4))
    data = data.astype('object')
    bad_data_idx = random((3, 4)) < 0.5
    data[bad_data_idx] = 'bad_data'
    # convert data to string stream
    astr = str(data).replace('[','').replace(']','')
    fname = StringIO( astr )
    data_type = int32
    no_data = 0
    out = loadtxt_nodata(fname, no_data=no_data, dtype=data_type)
    assert out.dtype == data_type
    data[bad_data_idx] = no_data
    assert all(out == data)
