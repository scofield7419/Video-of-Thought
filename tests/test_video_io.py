import os

import pytest
from PIL import Image

from vot.video import VideoClip, load_video, sample_timestamps, uniform_indices


def test_sample_timestamps():
    ts = sample_timestamps(10.0, fps=8, max_frames=None)
    assert len(ts) == 80 and abs(ts[1] - ts[0] - 0.125) < 1e-9 and ts[0] >= 0 and ts[-1] < 10
    capped = sample_timestamps(10.0, fps=8, max_frames=16)
    assert len(capped) == 16 and abs(capped[0] - 10 / 32) < 1e-9
    seg = sample_timestamps(10.0, fps=1, start=2.0, end=5.0)
    assert len(seg) == 3 and all(2.0 <= t < 5.0 for t in seg)
    assert len(sample_timestamps(0.2, fps=1)) == 1
    assert uniform_indices(10, 3) == [1, 5, 8] and uniform_indices(2, 5) == [0, 1]


def test_load_from_images_and_dir(tmp_path):
    imgs = [Image.new("RGB", (32, 24), (i, 0, 0)) for i in range(20)]
    clip = load_video(imgs, fps=1.0, max_frames=4, frames_fps=2.0)
    assert len(clip) == 4 and clip.size == (32, 24)
    for i, im in enumerate(imgs):
        im.save(tmp_path / f"{i:04d}.jpg")
    clip2 = load_video(str(tmp_path), fps=1.0, max_frames=5, frames_fps=2.0)
    assert len(clip2) == 5 and clip2.timestamps == sorted(clip2.timestamps)
    sub = VideoClip(imgs, [float(i) for i in range(20)]).subsample(3)
    assert len(sub) == 3 and sub.meta["kept_positions"] == [4, 11, 17]
    assert sub.resized(16).size == (16, 12)


def test_load_mp4(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        w.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    w.release()
    if not os.path.getsize(path):
        pytest.skip("no mp4 encoder available")
    for backend in ["opencv", "pyav"]:
        try:
            clip = load_video(path, fps=2.0, max_frames=None, backend=backend)
        except RuntimeError as e:
            if "not installed" in str(e):
                continue
            raise
        assert len(clip) == 6, backend
        assert clip.timestamps == sorted(clip.timestamps)
        assert abs(clip.duration - 3.0) < 0.2


def test_review_regressions_sampling(tmp_path):
    assert len(sample_timestamps(10, 1, 0)) == 10  # 0 = no cap
    for i in range(100):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i:04d}.jpg")
    clip = load_video(str(tmp_path), fps=1.0, max_frames=4, start=10, end=20, frames_fps=1.0)
    assert all(10 <= t < 20 for t in clip.timestamps)
