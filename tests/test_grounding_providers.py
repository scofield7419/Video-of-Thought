from PIL import Image

from vot.backends.mock import ScriptedLLM
from vot.grounding import StaticSTSGProvider, resample_to_clip
from vot.grounding.detector_parser import DetectorSTSGParser
from vot.grounding.vlm_parser import VLMSTSGParser
from vot.stsg import STSG
from vot.video import VideoClip


def _clip(n, size=(200, 100)):
    return VideoClip([Image.new("RGB", size) for _ in range(n)], [float(i) for i in range(n)])


def test_detector_parser_tracks_and_relations():
    # two men moving right, one dog static; boxes in pixels (xyxy) on a 200x100 frame
    def detect(img, vocab):
        k = detect.k
        detect.k += 1
        return [
            ("the man", 0.9, (10 + 5 * k, 10, 30 + 5 * k, 60)),
            ("the man", 0.8, (120 + 5 * k, 10, 140 + 5 * k, 60)),
            ("dog", 0.7, (60, 60, 80, 80)),
            ("dog", 0.3, (61, 61, 80, 80)),  # duplicate removed by NMS
        ]

    detect.k = 0
    p = DetectorSTSGParser(detector=detect, extra_vocabulary=["dog"])
    g = p(_clip(4), targets=["the man"])
    assert len(g.frames) == 4
    ids = g.frame(1).object_ids()
    assert sorted(ids) == ["dog", "man-1", "man-2"]
    tracks = g.tracks()
    assert len(tracks) == 3 and all(len(v) == 4 for v in tracks.values())
    m1 = [o.bbox[0] for _, o in tracks["man-1"]]
    assert m1 == sorted(m1)
    assert g.frame(1).triplets and all(t.subject in ids and t.object in ids for t in g.frame(1).triplets)
    g.validate()


def test_vlm_parser_chunks_and_renumbers():
    def fn(msgs):
        n = len([p for p in msgs[-1].content if hasattr(p, "clip")][0].clip)
        frames = "\n".join(
            f'Frame {k}: {{Objects: [{{"dog": [{k}.0, 10, 5, 5]}}], Triplets: []}}' for k in range(1, n + 1)
        )
        return frames

    llm = ScriptedLLM(fn)
    g = VLMSTSGParser(llm, chunk_size=3)(_clip(7), targets=["dog"])
    assert g.frame_indices() == list(range(1, 8))
    assert len(llm.calls) == 3
    assert "Previous frame" in llm.calls[1][-1].text()
    assert len(g.tracks()) == 1  # coref inferred across chunk borders
    assert g.frame(5).timestamp == 4.0


def test_static_provider_and_resample(tmp_path, truck_stsg):
    import json

    (tmp_path / "vid1.json").write_text(json.dumps(truck_stsg.to_dict()))
    p = StaticSTSGProvider(str(tmp_path))
    assert p(_clip(3), video_id="vid1").to_dict() == truck_stsg.to_dict()
    assert p(_clip(3), video_id="missing").meta["missing"]

    dense = STSG.from_frames([{"objects": {"a": [i, 0, 1, 1]}, "timestamp": i * 0.25} for i in range(12)])
    clip = VideoClip([Image.new("RGB", (8, 8))] * 3, [0.0, 1.0, 2.5])
    r = resample_to_clip(dense, clip)
    assert r.frame_indices() == [1, 2, 3] and r.frame(3).get("a").bbox[0] == 10.0 and len(r.coref) == 2
