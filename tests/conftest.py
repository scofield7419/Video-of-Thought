import os
import sys

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def truck_stsg():
    from vot.stsg import STSG

    return STSG.from_frames(
        [
            {"objects": {"truck": [10, 40, 30, 25], "man-1": [45, 45, 6, 20], "trash can": [55, 55, 6, 10], "street": [0, 60, 100, 40]},
             "triplets": [("truck", "on", "street"), ("man-1", "next to", "trash can")]},
            {"objects": {"truck": [20, 40, 30, 25], "man-1": [48, 45, 6, 20], "trash can": [50, 50, 6, 10], "street": [0, 60, 100, 40]},
             "triplets": [("truck", "on", "street"), ("man-1", "holding", "trash can")]},
            {"objects": {"truck": [30, 40, 30, 25], "man-1": [40, 45, 6, 20], "trash can": [42, 40, 6, 10], "street": [0, 60, 100, 40]},
             "triplets": [("man-1", "pouring", "trash can"), ("trash can", "above", "truck")]},
        ]
    )


@pytest.fixture
def clip3():
    from PIL import Image

    from vot.video import VideoClip

    return VideoClip([Image.new("RGB", (320, 240), (40 * i, 90, 120)) for i in range(3)], [0.5, 1.5, 2.5])


TRUCK_Q = "What is the relationship between the white truck and this neighborhood?"
TRUCK_OPTS = ["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"]
