# Training MotionEpic

> **Status.** MotionEpic weights are **not released**. This training stack is provided so that the
> model can be reproduced; it follows the paper (Sec. 3 and Sec. 5, "Implementations") and is
> covered by unit tests with tiny random configurations (`tests/test_motionepic_model.py`,
> `tests/test_stsg_encoder.py`, `tests/test_sft_data.py`, `tests/test_grounding_tasks.py`).
> It has not been re-run at full scale in this repository, so expect to tune batch sizes and
> learning rates for your hardware. The VoT reasoning framework itself does not need these weights:
> it runs on API models (see [api_version.md](api_version.md)).

## 1. Architecture → code

| Paper component | Setting in the paper | Code |
|---|---|---|
| Backbone LLM | Vicuna-7B v1.5, frozen, LoRA-tuned | `MotionEpicForCausalLM` (`vot/models/motionepic/modeling.py`), `apply_lora` (`builder.py`) |
| Video encoder | CLIP ViT-L/14, frozen, per frame | `CLIPVisionTower.frame_features` (`vision.py`); frames are padded to a square before CLIP preprocessing (no centre crop), so every full-frame STSG box stays visible |
| Video projector | Q-Former | `VideoQFormerProjector` (`projector.py`): frame-level BLIP-2 Q-Former → temporal position embeddings → video-level Q-Former → linear |
| STSG node `v = (c, f, b)` | `f` from CLIP, `b = (x, y, w, h)` | `STSGFeaturizer` crops one RoI per node (`featurizer.py`); `CLIPVisionTower.image_embeds` gives `f`; `PhraseEncoder` embeds `c` and predicates |
| STSG encoder | Graph Transformer with recurrent propagation, 6 layers, 768-d | `RecurrentGraphTransformer` (`stsg_encoder.py`) |
| Temporal coreference edges | link the same object in frames `k-1 → k` | `coref_prev` pointers; GRU state passing in `RecurrentGraphTransformer.forward` |
| STSG expression generation | text output, recycled as next-round input | `vot.stsg.expression` (format/parse); `MotionEpicBackend` feeds a generated partial STSG back as `<stsg>` graph tokens + text |
| Frame sampling | 8 fps | `MotionEpicConfig.sample_fps = 8.0`, capped by `max_frames = 32` |

**STSG encoder in detail.** For every frame `t`, node inputs are
`x = W_f f + W_c c + MLP(box) + time(t)` (a learned embedding replaces `W_f f` when a node has no
box). The node state entering frame `t` is `GRU(x_{t,i}, h_{t-1, coref(i)})`, with a zero state
when node `i` starts a new track. Six edge-featured graph-transformer layers (Dwivedi & Bresson,
2020) then run inside the frame graph: attention is restricted to neighbours plus a self loop, and
each predicate edge is used in both directions with a learned direction embedding. A Perceiver
resampler pools all node states into `stsg_num_tokens` (default 32) LLM tokens. Placeholders
`<video>` and `<stsg>` become negative sentinel ids, so the LLM vocabulary is unchanged.

**What a checkpoint contains.** Only what training changes is saved:

```
checkpoints/motionepic-stage3/
├── motionepic_config.json      # MotionEpicConfig (incl. base LLM / CLIP ids)
├── mm_trainables.safetensors   # video_projector.* + stsg_encoder.*
└── lora/                       # PEFT LoRA adapter of Vicuna
```

## 2. Data preparation

All stages read one record format (`vot/data/sft.py`):

```json
{"id": "001YG-L3", "frames": ["001YG.mp4/000089.png", "..."], "frame_times": [3.0, 3.5],
 "stsg": {"frames": ["..."]},
 "conversations": [{"from": "human", "value": "<video>\n..."}, {"from": "gpt", "value": "..."}]}
```

Frames are selected by priority `frames` > `frame_times` > uniform sampling of `video` at
`sample_fps`. Grounding records always pin the exact frames, so that "Frame k" in a target STSG is
the k-th frame the model sees. STSG nodes are matched to frames by timestamp only when the clip has
real timestamps (`frame_times` or a decoded video); a `frames` list without `frame_times` is
aligned by index.

| Corpus | Used in | How to prepare |
|---|---|---|
| WebVid captions | stage 1 | `python scripts/data/convert_sft_corpora.py webvid --src results_10M_train.csv --out data/sft/webvid_caption.jsonl` |
| Action Genome STSGs (~10K videos, manual) | stage 2 | (1) download AG annotations and dump frames with the [AG toolkit](https://github.com/JingweiJ/ActionGenome); (2) `python scripts/data/convert_action_genome.py --ag_root data/action_genome --videos_dir data/charades/videos --split train --out data/action_genome/stsg_train.jsonl` (keep AG test videos out of training) |
| WebVid STSGs (350K videos with clear actions) | stage 2 | Parse with an off-the-shelf dynamic scene-graph parser (the paper uses AP-Net, Li et al., CVPR 2022). **The parser is external to this repo.** Write one JSONL row per video: `{"video_id", "video", "frame_times", "stsg"}` (`STSG.to_dict()`, boxes in percent) |
| Grounding corpus L1–L5 | stage 2 | `python scripts/data/build_grounding_corpus.py --stsg data/action_genome/stsg_train.jsonl --out data/sft/grounding_ag.jsonl --charades_csv Charades_v1_train.csv --charades_classes Charades_v1_classes.txt` (and the same for the WebVid STSGs, without Charades). Rows with `"split": "test"` are skipped unless `--include_test` is passed |
| VideoChat / Video-ChatGPT instructions | stage 3 | `python scripts/data/convert_sft_corpora.py videochat --src videochat_instruct_11k.json --out data/sft/videochat_instruct.jsonl`; `... videochatgpt --src VideoInstruct100K.json --out data/sft/videochatgpt_instruct.jsonl` |

**Grounding objectives** (`vot/data/grounding_tasks.py`, paper Sec. 3.3):

| | Input | Target |
|---|---|---|
| L1 | video + STSG as `<stsg>` graph tokens | "Yes." / "No." (negatives: STSG of another video, or the same STSG with swapped categories, swapped predicates or reversed time) |
| L2 | video | full STSG expression |
| L3 | video + action description(s) | partial STSG of the involved objects |
| L4 | video + key object(s) | temporal action description + tracklet(s) |
| L5 | video + box of an object in frame k | label + tracklet |

Action descriptions come from Charades action labels (with time spans) when available, otherwise
from contiguous spans of relation triplets in the STSG (e.g. "the person is holding the cup from
Frame 2 to Frame 5"). Each task has three paraphrased instructions; sampling is deterministic given
`--seed`. Time-reversal negatives of L1 keep every frame slot's timestamp, so the only signal is
the content.

**Sequence length.** A full-video STSG expression grows quickly with the number of frames (every
digit of a box is a LLaMA token), so L2 samples use at most 8 frames (`--l2_max_frames`) and the
other tasks 16 (`--max_frames`). `--max_tokens` (default 3800 for a 4096 context) bounds the text of
each sample with a conservative estimate (or exact counts with `--tokenizer`); over-long L2
samples are rebuilt with fewer frames and other over-long samples are dropped. At training time
`SFTCollator` budgets the spliced length (each `<video>` / `<stsg>` expands to 32 tokens): a reply
cut by truncation is masked entirely, so the model is never trained to stop mid-graph without
`</s>`, and samples left without supervision are dropped from the batch.

## 3. Stages

| Stage | Data | Trainable | Key settings | Command |
|---|---|---|---|---|
| 1. Video pretraining | WebVid captions | Q-Former projector | lr 1e-4, BLIP-2 Q-Former init | `bash scripts/train/stage1_pretrain.sh` |
| 2. Grounding-aware tuning | L1–L5 on AG + WebVid STSGs | projector, STSG encoder, LoRA (r=128, α=256) | LoRA lr 2e-4, `mm_lr` 1e-4, 4096 tokens | `bash scripts/train/stage2_grounding.sh` |
| 3. Instruction tuning | VideoChat + Video-ChatGPT | projector, STSG encoder, LoRA (continued) | LoRA lr 2e-4, `mm_lr` 2e-5 | `bash scripts/train/stage3_instruct.sh` |
| 4. Task fine-tuning (optional) | benchmark train split | same | fine-tuning setting of Tables 1–3 | `bash scripts/train/stage4_task.sh` |

Throughout all stages CLIP and the LLM weights stay frozen (the LLM is adapted only via LoRA), as
in the paper. Each stage reads its YAML (`configs/train/stage*.yaml`) and starts from the previous
stage's checkpoint via `init_from`. Any value can be overridden on the command line:

```bash
NUM_GPUS=8 DS_CONFIG=configs/deepspeed/zero2.json \
  bash scripts/train/stage2_grounding.sh --per_device_train_batch_size 2 --gradient_accumulation_steps 8
```

The scripts call `torchrun -m vot.train.train --config <yaml> --deepspeed <json>`. The paper
trained on 16 A100 GPUs.

**ZeRO-2 only.** DeepSpeed ZeRO-3 is rejected at start-up. Different ranks run different
data-dependent branches (video and STSG encoders only for samples that have them, RoI embeddings
only for boxed nodes), which breaks ZeRO-3's collective parameter gathering. BLIP-2 initialisation
and LoRA merging also assume unpartitioned weights. The LLM and CLIP are frozen, so ZeRO-2
(`configs/deepspeed/zero2.json`) fits a 7B backbone on 40 GB GPUs; lower
`per_device_train_batch_size` and raise `gradient_accumulation_steps` if needed. Without DeepSpeed
(plain HF Trainer), trainable parameters are kept in fp32 so that bf16 runs still apply small
AdamW updates.
Use `--attn_implementation flash_attention_2` if FlashAttention is installed.

## 4. Using a trained checkpoint in VoT

```python
from vot import VideoOfThought, VoTConfig, build_backend

backend = build_backend("motionepic:checkpoints/motionepic-stage3")   # MotionEpicBackend
vot = VideoOfThought(backend, VoTConfig.from_yaml("configs/vot/motionepic.yaml"))
result = vot("demo.mp4", "What is the relationship between the white truck and this neighborhood?",
             ["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"])
print(result.pretty())
```

`MotionEpicBackend(checkpoint, llm_name_or_path=None, vision_name_or_path=None, device=None,
torch_dtype="bfloat16", stsg_mode="graph+text")` loads the base Vicuna and CLIP weights named in
`motionepic_config.json` (override them to point at local copies), the multimodal weights and the
LoRA adapter (merged for inference). In VoT, the partial STSG that MotionEpic generates in Step 2
is parsed and passed back in Steps 3 and 5 as an `STSGPart`. The backend encodes it with the STSG
encoder (RoIs cropped from the same sampled frames) and also includes its text expression
(`stsg_mode="graph+text"`). Keep the VoT sampling at `max_frames <= 32` so that STSG frame numbers
match the frames the model sees.

## 5. Debugging on a laptop

The tests build a tiny random MotionEpic (2-layer LLaMA, 2-layer CLIP, hash phrase encoder), so the
full pipeline (featurization → splicing → loss → `generate` → save/load → LoRA) runs on CPU in
seconds:

```bash
python -m pytest tests/test_stsg_encoder.py tests/test_motionepic_model.py tests/test_sft_data.py tests/test_grounding_tasks.py -q
```
