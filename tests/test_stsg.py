from vot.stsg import STSG, CorefLink, FrameGraph, SGObject, Triplet, ground_targets, match_targets, parse_stsg
from vot.stsg.geometry import iou, percent_to_pixel, pixel_to_percent, spatial_predicate
from vot.stsg.ops import align_to_reference, neighbor_scene, union

PAPER_EXPR = """Frame 1: {
    Objects: [{"car-1": [0.0,13.4,7.0,8.1]}, {"car-2": [27.0,12.9,9.0,4.0]},
              {"house": [15.0,0.0,34.5,17.0]}, {"street": [9.0,13.5,16.5,15.0]}],
    Triplets: [("car-1", "on the left", "street"), ("car-2", "on the right", "street"), ("house",
               "beside", "street")]
}
Frame 2: {
    Objects: [{"car": [9.0,2.4,33.7,25.6]}, {"man-1": [1.8,16.4,4.2,10.0]},
              {"house": [43.0,14.4, 6.2,12.0]}, {"street": [17.9,0.0,10.2,10.0]}],
    Triplets: [("man-1", "beside", "car"), ("car", "on", "street")]
}
Temporal Coreference Edge: [
    {Frame 1 -> 2: [("house", "house"), ("street", "street"), ("car-1", "car")]},
]"""


def test_parse_paper_expression():
    g = parse_stsg(PAPER_EXPR)
    assert [f.index for f in g.frames] == [1, 2]
    assert g.frame(1).get("car-2").bbox == (27.0, 12.9, 9.0, 4.0)
    assert Triplet("house", "beside", "street") in g.frame(1).triplets
    assert CorefLink(1, "car-1", 2, "car") in g.coref
    g.validate()
    tracks = g.tracks()
    assert [(k, o.id) for k, o in tracks["car-1"]] == [(1, "car-1"), (2, "car")]


def test_roundtrip(truck_stsg):
    for compact in (False, True):
        g2 = parse_stsg(truck_stsg.to_expression(compact=compact))
        assert g2.to_dict() == truck_stsg.to_dict()
    assert STSG.from_json(truck_stsg.to_json()).to_dict() == truck_stsg.to_dict()


def test_tolerant_parsing():
    noisy = """Sure! The partial STSG in tracking [the car] is:
```
Frame 1: {Objects: [{'car-1': [0, 13.4, 7, 8.1]}, {"street": [9.0,13.5,16.5,15.0]}], Triplets: [["car-1", "on the left", "street"], ("car-1","near","tree")]}
Frame 3 (t=2.0s): {Objects: [{“car-1”: [3,13,7,8]}], Triplets: []}
```"""
    g = parse_stsg(noisy)
    assert g.frame(1).get("tree") is not None and g.frame(1).get("tree").bbox is None  # repaired node
    assert g.frame(3).get("car-1").bbox == (3.0, 13.0, 7.0, 8.0)
    assert CorefLink(1, "car-1", 3, "car-1") in g.coref  # inferred, no coref section
    assert parse_stsg("no graph here").is_empty


def test_json_variant_and_pixels():
    js = '{"frames": [{"index": 1, "objects": {"dog": [10, 20, 30, 40]}, "triplets": []}]}'
    assert parse_stsg(js).frame(1).get("dog").bbox == (10.0, 20.0, 30.0, 40.0)
    g = parse_stsg('Frame 1: {Objects: [{"dog": [64, 48, 128, 96]}], Triplets: []}', pixel_size=(640, 480))
    assert g.frame(1).get("dog").bbox == (10.0, 10.0, 20.0, 20.0)


def test_box_unit_autodetection():
    from vot.stsg.expression import detect_box_units

    px = 'Frame 1: {Objects: [{"dog": [64, 48, 128, 96]}, {"cat": [320, 240, 10, 10]}], Triplets: []}'
    unit = 'Frame 1: {Objects: [{"dog": [0.1, 0.1, 0.2, 0.2]}], Triplets: []}'
    assert detect_box_units(px) == "pixel" and detect_box_units(unit) == "unit"
    assert parse_stsg(px, image_size=(640, 480)).frame(1).get("dog").bbox == (10.0, 10.0, 20.0, 20.0)
    assert parse_stsg(unit, image_size=(640, 480)).frame(1).get("dog").bbox == (10.0, 10.0, 20.0, 20.0)
    pct = 'Frame 1: {Objects: [{"dog": [10, 10, 20, 20]}], Triplets: []}'
    assert parse_stsg(pct, image_size=(640, 480)).frame(1).get("dog").bbox == (10.0, 10.0, 20.0, 20.0)


def test_subgraph_hops_and_tracks(truck_stsg):
    g0 = truck_stsg.subgraph(["man-1"], hops=0)
    assert all(f.object_ids() == ["man-1"] for f in g0.frames) and len(g0.frames) == 3
    g1 = truck_stsg.subgraph(["man-1"], hops=1)
    assert set(g1.frame(2).object_ids()) == {"man-1", "trash can"}
    g2 = truck_stsg.subgraph(["man-1"], hops=2)
    assert "truck" in g2.frame(3).object_ids()
    assert len(g1.coref) == 4  # man-1 and trash can over two transitions


def test_follow_tracks_across_renamed_nodes():
    g = STSG(
        frames=[FrameGraph(1, [SGObject("man", (0, 0, 1, 1))]), FrameGraph(2, [SGObject("man-1", (1, 0, 1, 1))])],
        coref=[CorefLink(1, "man", 2, "man-1")],
    )
    sub = g.subgraph(["man"])
    assert sub.frame(2) is not None and sub.frame(2).object_ids() == ["man-1"]


def test_select_frames_relinks(truck_stsg):
    s = truck_stsg.select_frames([1, 3])
    assert CorefLink(1, "truck", 3, "truck") in s.coref
    assert len(truck_stsg.subsample(2)) == 2 and truck_stsg.subsample(2).frame_indices() == [1, 3]


def test_matching(truck_stsg):
    m = match_targets(["the white truck", "the garbage can", "the neighborhood street", "the man-1", "a person"], truck_stsg)
    assert m["the white truck"] == ["truck"]
    assert m["the neighborhood street"] == ["street"]
    assert m["the man-1"] == ["man-1"]
    assert m["a person"] == ["man-1"]  # synonym group
    assert not ground_targets(["a bicycle"], truck_stsg).frames


def test_ops(truck_stsg):
    partial = parse_stsg('Frame 2: {Objects: [{"man-1": [0, 0, 1, 1]}], Triplets: []}')
    aligned = align_to_reference(partial, truck_stsg)
    assert aligned.frame(2).get("man-1").bbox == truck_stsg.frame(2).get("man-1").bbox
    scene = neighbor_scene(aligned, truck_stsg, hops=1)
    assert set(scene.frame(2).object_ids()) == {"man-1", "trash can"}
    u = union(aligned, truck_stsg.subgraph(["truck"]))
    assert {"man-1", "truck"} <= set(u.frame(2).object_ids())


def test_geometry():
    a = (10.0, 10.0, 20.0, 20.0)
    assert abs(iou(a, a) - 1) < 1e-9 and iou(a, (50, 50, 5, 5)) == 0
    assert pixel_to_percent(percent_to_pixel(a, 640, 480), 640, 480) == a
    assert spatial_predicate((0, 0, 10, 10), (50, 0, 10, 10)) == "on the left of"
    assert spatial_predicate((12, 12, 2, 2), (10, 10, 20, 20)) == "inside"


def test_review_regressions_parser():
    bold = '**Frame 1**: {Objects: [{"man": [1,2,3,4]}], Triplets: []}\n**Frame 2**: {Objects: [{"man": [2,2,3,4]}], Triplets: []}'
    assert parse_stsg(bold).frame_indices() == [1, 2]
    assert parse_stsg('### Frame 1\n{Objects: [{"man": [1,2,3,4]}], Triplets: []}').frame_indices() == [1]
    pre = ("Here are the scene graphs and the temporal coreference edges of [the man]:\n"
           'Frame 1: {Objects: [{"man": [1,2,3,4]}], Triplets: []}\nFrame 2: {Objects: [{"man-1": [2,2,3,4]}], Triplets: []}\n'
           'Temporal Coreference Edge: [{Frame 1 -> 2: [("man", "man-1")]}]')
    g = parse_stsg(pre)
    assert g.frame_indices() == [1, 2] and CorefLink(1, "man", 2, "man-1") in g.coref
    js = '{"frames":[{"frame":"Frame 1","objects":[{"man-1":[64,48,128,96]}],"triplets":[{"subject":"man-1","predicate":"on","object":"road"}]}]}'
    g = parse_stsg(js, image_size=(640, 480))
    assert g.frame(1).get("man-1").bbox == (10.0, 10.0, 20.0, 20.0) and Triplet("man-1", "on", "road") in g.frame(1).triplets
    assert parse_stsg('{"frames": [{"frame": 1, "objects": "bad"}]}').num_nodes == 0


def test_select_frames_keeps_non_adjacent_coref():
    g = STSG(
        frames=[FrameGraph(1, [SGObject("man", (0, 0, 1, 1))]), FrameGraph(2, [SGObject("dog", (0, 0, 1, 1))]),
                FrameGraph(3, [SGObject("person", (1, 0, 1, 1))])],
        coref=[CorefLink(1, "man", 3, "person")],
    )
    assert CorefLink(1, "man", 3, "person") in g.select_frames([1, 2, 3]).coref
    assert len(g.select_frames([1, 2, 3]).tracks()) == 2
