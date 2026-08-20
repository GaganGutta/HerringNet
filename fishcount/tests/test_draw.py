import numpy as np

from fishcount.draw import annotate
from helpers import fish


def test_annotate_returns_marked_copy_and_leaves_input_untouched() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    before = image.copy()

    out = annotate(image, [fish()])

    assert out is not image
    assert np.array_equal(image, before)
    assert out.shape == image.shape
    assert out.any()  # a box was drawn


def test_annotate_with_no_detections_is_a_plain_copy() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    out = annotate(image, [])
    assert out is not image
    assert not out.any()  # no banner, no boxes, nothing drawn
