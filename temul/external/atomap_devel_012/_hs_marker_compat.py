import numpy as np
from hyperspy.misc.utils import isiterable
from hyperspy.utils.markers import Lines, Points, Texts


def _is_sequence(value):
    return isiterable(value) and not isinstance(value, (str, bytes))


def _point_offsets(x, y):
    if _is_sequence(x) or _is_sequence(y):
        x_values = list(x) if _is_sequence(x) else [x] * len(y)
        y_values = list(y) if _is_sequence(y) else [y] * len(x_values)
        if len(x_values) != len(y_values):
            raise ValueError("x and y must have the same length")
        offsets = np.empty(len(x_values), dtype=object)
        for i, (x_value, y_value) in enumerate(zip(x_values, y_values)):
            offsets[i] = np.array([[x_value, y_value]], dtype=float)
        return offsets
    return np.array([[x, y]], dtype=float)


def Point(x, y, size=20, **kwargs):
    return Points(offsets=_point_offsets(x, y), sizes=size, **kwargs)


def _line_segments(x1, y1, x2, y2):
    if any(_is_sequence(value) for value in (x1, y1, x2, y2)):
        lengths = [
            len(value) for value in (x1, y1, x2, y2)
            if _is_sequence(value)
        ]
        length = lengths[0]
        if any(item != length for item in lengths):
            raise ValueError("Line coordinates must have the same length")
        coords = []
        for value in (x1, y1, x2, y2):
            coords.append(list(value) if _is_sequence(value) else [value] * length)
        segments = np.empty(length, dtype=object)
        for i, values in enumerate(zip(*coords)):
            segments[i] = np.array(
                [[[values[0], values[1]], [values[2], values[3]]]],
                dtype=float,
            )
        return segments
    return np.array([[[x1, y1], [x2, y2]]], dtype=float)


def LineSegment(x1, y1, x2, y2, **kwargs):
    return Lines(segments=_line_segments(x1, y1, x2, y2), **kwargs)


def Text(x, y, text, size=20, **kwargs):
    text_kwargs = dict(kwargs)
    if "color" in text_kwargs and "facecolors" not in text_kwargs:
        text_kwargs["facecolors"] = text_kwargs.pop("color")
    text_kwargs["sizes"] = size
    text_kwargs["texts"] = text
    return Texts(offsets=_point_offsets(x, y), **text_kwargs)
