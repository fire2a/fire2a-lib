#!/usr/bin/env python3
"""Helper classes for transforming c2f lookup tables rgb to QGIS qml format"""
from colorsys import hls_to_rgb, rgb_to_hls

import numpy as np
from pandas import read_csv


def rgb2hex_color(r, g, b) -> str:
    """int 0-255 rgb to hex"""
    return "#%02x%02x%02x" % (r, g, b)
    # return '%02x%02x%02x'%(int(r*255), int(g*255), int(b*255))


def fuel_lookuptable_colorconvert(afile="spain_lookup_table.csv"):
    df = read_csv(afile, usecols=["grid_value", "r", "g", "b", "h", "s", "l"], dtype=np.int16)

    for t in df.itertuples():
        print((t.r, t.g, t.b), hls_to_rgb(t.h, t.l, t.s), rgb_to_hls(t.r, t.g, t.b), (t.h, t.l, t.s))
        assert np.all(rgb_to_hls(t.r, t.g, t.b) == (t.h, t.l, t.s)) and np.all(
            (t.r, t.g, t.b) == hls_to_rgb(t.h, t.l, t.s)
        )


def fuel_lookuptable_csv2qml(afile="kitral_lookup_table.csv", qmlfile="qml_lookup_table.qml"):
    """convert csv to qml colorPalette section
    printing results to stdout
    TODO modify a qml file <colorPallete> section
    columns : ['grid_value', 'export_value', 'descriptive_name', 'fuel_type', 'r', 'g', 'b', 'h', 's', 'l']
    """
    df = read_csv(afile)
    # print('<paletteEntry color="#ffffff" label="NA" alpha="0" value="0"/>')
    for row in df.itertuples():
        hex_color = rgb2hex_color(row.r, row.g, row.b)
        print(f'<paletteEntry color="{hex_color}" label="{row.fuel_type}" alpha="255" value="{row.grid_value}"/>')
