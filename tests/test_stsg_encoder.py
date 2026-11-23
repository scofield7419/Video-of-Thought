import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from _tiny_motionepic import kitchen_stsg, tiny_clip, tiny_config  # noqa: E402

from vot.models.motionepic.featurizer import STSGFeaturizer  # noqa: E402
from vot.models.motionepic.stsg_encoder import GraphTransformerLayer, RecurrentGraphTransformer  # noqa: E402
from vot.stsg import STSG  # noqa: E402


def _inputs(b=2, t=3, n=4, f=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        roi_feat=torch.randn(b, t, n, f, generator=g),
        has_roi=torch.ones(b, t, n, dtype=torch.bool),
        label_feat=torch.randn(b, t, n, f, generator=g),
        node_box=torch.rand(b, t, n, 4, generator=g) * 0.5,
        node_mask=torch.ones(b, t, n, dtype=torch.bool),
        frame_time=torch.arange(t).float().expand(b, t).clone(),
        frame_mask=torch.ones(b, t, dtype=torch.bool),
        edge_index=torch.zeros(0, 4, dtype=torch.long),
        edge_feat=torch.zeros(0, f),
        coref_prev=torch.full((b, t, n), -1, dtype=torch.long),
    )


def test_gt_layer_respects_adjacency():
    torch.manual_seed(0)
    layer = GraphTransformerLayer(32, 4).eval()
    h, e = torch.randn(1, 3, 32), torch.randn(1, 3, 3, 32)
    adj = torch.eye(3, dtype=torch.bool)[None].clone()
    adj[0, 0, 1] = True  # node 0 receives from node 1
    out1, _ = layer(h, e, adj)
    h2 = h.clone()
    h2[0, 2] += 5.0  # node 2 is isolated from nodes 0/1
    out2, _ = layer(h2, e, adj)
    assert torch.allclose(out1[0, :2], out2[0, :2], atol=1e-6)
    h3 = h.clone()
    h3[0, 1] += 5.0
    out3, _ = layer(h3, e, adj)
    assert not torch.allclose(out1[0, 0], out3[0, 0])


def test_encoder_shapes_and_padding():
    cfg = tiny_config()
    enc = RecurrentGraphTransformer(cfg, llm_hidden_size=48).eval()
    x = _inputs()
    tokens, states = enc(**x)
    assert tokens.shape == (2, cfg.stsg_num_tokens, 48)
    assert states.shape == (2, 3, 4, cfg.stsg_hidden_size)
    x["node_mask"][1, :, 2:] = False
    _, states = enc(**x)
    assert torch.all(states[1, :, 2:] == 0)
    empty = _inputs(b=1)
    empty["node_mask"][:] = False
    tok, _ = enc(**empty)
    assert torch.isfinite(tok).all()


def test_recurrence_follows_coref_only():
    cfg = tiny_config()
    enc = RecurrentGraphTransformer(cfg, llm_hidden_size=48).eval()
    x = _inputs(b=1, t=2, n=2)
    x["coref_prev"][0, 1, 0] = 0  # frame-2 node 0 continues frame-1 node 0; node 1 starts a new track
    _, s1 = enc(**x)
    y = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    y["roi_feat"][0, 0] += 3.0  # change frame-1 inputs only
    _, s2 = enc(**y)
    assert not torch.allclose(s1[0, 1, 0], s2[0, 1, 0])  # linked node changes
    assert torch.allclose(s1[0, 1, 1], s2[0, 1, 1], atol=1e-6)  # unlinked, unconnected node does not
    y["coref_prev"][0, 1, 0] = -1
    _, s3 = enc(**y)
    _, s4 = enc(**{**x, "coref_prev": y["coref_prev"]})
    assert torch.allclose(s3[0, 1], s4[0, 1], atol=1e-6)  # no coref: frame 2 independent of frame 1


def test_edges_change_states():
    cfg = tiny_config()
    enc = RecurrentGraphTransformer(cfg, llm_hidden_size=48).eval()
    x = _inputs(b=1, t=1, n=3)
    _, s0 = enc(**x)
    x["edge_index"] = torch.tensor([[0, 0, 0, 1]])
    x["edge_feat"] = torch.randn(1, 16)
    _, s1 = enc(**x)
    assert not torch.allclose(s0[0, 0, 0], s1[0, 0, 0])
    assert torch.allclose(s0[0, 0, 2], s1[0, 0, 2], atol=1e-6)


def test_featurizer_layout_and_coref():
    from vot.models.motionepic.processor import default_image_processor

    feat = STSGFeaturizer(default_image_processor(32), max_frames=8, max_nodes=6)
    g = kitchen_stsg()
    clip = tiny_clip(3)
    x = feat.encode(g, clip)
    assert x.node_mask.shape == (3, 3)
    assert x.node_mask[2].sum() == 2
    assert x.roi_pixels is not None and x.roi_pixels.shape[0] == 8
    pos = {oid: i for i, oid in enumerate(x.node_ids[1])}
    prev = {oid: i for i, oid in enumerate(x.node_ids[0])}
    assert x.coref_prev[1, pos["cup"]] == prev["cup"]
    assert len(x.edges) == 4 and set(x.predicates) == {"looking at", "on", "holding"}
    assert torch.allclose(x.frame_time, torch.tensor([0.0, 0.5, 1.0]))

    no_clip = feat.encode(g, None)
    assert (no_clip.node_roi == -1).all() and no_clip.roi_pixels is None

    batch = STSGFeaturizer.collate([x, feat.encode(STSG.from_frames([{"objects": {"dog": [0, 0, 10, 10]}}]), clip)])
    assert batch.node_mask.shape == (2, 3, 3)
    assert batch.frame_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.roi_pixels.shape[0] == 9
    assert batch.node_roi[1, 0, 0] == 8
    assert batch.phrases[batch.node_label[1, 0, 0]] == "dog"
    assert batch.edge_index[:, 0].eq(0).all()


def test_featurizer_truncates_nodes_keeping_related_ones():
    feat = STSGFeaturizer(None, max_frames=8, max_nodes=2)
    g = STSG.from_frames([{"objects": {"a": [0, 0, 5, 5], "b": [10, 10, 5, 5], "c": [20, 20, 5, 5]}, "triplets": [("a", "near", "c")]}])
    x = feat.encode(g)
    assert x.node_ids[0] == ["a", "c"]
