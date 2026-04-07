import numpy as np

import temul.signal_plotting as sp


def test_get_cropping_area_applies_offset():
    area = sp.get_cropping_area([[10, 20], [30, 40]], crop_offset=5)

    assert area == (5, 35, 15, 45)


def test_color_palettes_known_and_unknown():
    zesty = sp.color_palettes('zesty')
    r_safe = sp.color_palettes('r_safe')

    assert len(zesty) == 4
    assert len(r_safe) == 12
    assert sp.color_palettes('unknown') == 'This option is not allowed.'


def test_rgb_to_dec_and_hex_to_rgb_round_trip():
    rgb_values = sp.hex_to_rgb(['#F5793A', '#0F2080'])
    dec_values = sp.rgb_to_dec(rgb_values)

    assert rgb_values == [(245, 121, 58), (15, 32, 128)]
    assert dec_values[0] == tuple(value / 256 for value in rgb_values[0])


def test_expand_palette_repeats_entries():
    palette = ['a', 'b', 'c']
    expanded = sp.expand_palette(palette, [1, 3, 2])

    assert expanded == ['a', 'b', 'b', 'b', 'c', 'c']


def test_get_polar_2d_colorwheel_color_list_length_and_range():
    u = np.array([1.0, 0.0, -1.0, 0.0])
    v = np.array([0.0, 1.0, 0.0, -1.0])

    colors = sp.get_polar_2d_colorwheel_color_list(u, v)

    assert len(colors) == len(u)
    assert all(len(color) == 3 for color in colors)
    assert np.all((np.asarray(colors) >= 0) & (np.asarray(colors) <= 1))
