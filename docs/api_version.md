# Running VoT with API models

The reasoning framework is model-agnostic: `VideoOfThought` talks to any
`VideoLLM` backend. This page covers the API version, which needs no training.

## Backends

| Spec | Backend | Notes |
|---|---|---|
| `openai:<model>` | `OpenAICompatibleBackend` | OpenAI API (GPT-4o, GPT-4.1, GPT-5, o-series). Reasoning models automatically get `max_completion_tokens` and no sampling parameters. |
| `openai:<model> --base-url <url>` | same | Any OpenAI-compatible server: vLLM / SGLang / LMDeploy serving Qwen2.5-VL, InternVL, LLaVA-Video; DashScope; Gemini's OpenAI endpoint. |
| `anthropic:<model>` | `AnthropicBackend` | Claude (default `claude-opus-5-5`). Sampling parameters are not sent (rejected by current models); `effort` (default `medium`) controls reasoning depth; a prompt-cache breakpoint is placed on the video frames, which are the shared prefix of Steps 1–3 and 5; server-side refusal fallback is enabled by default (`fallbacks="default"`, disable with `--backend-arg fallbacks=null`). |
| `motionepic:<ckpt>` | `MotionEpicBackend` | Local trained MotionEpic (see [training.md](training.md)). |
| `mock` | `RuleBasedMockLLM` | Offline stand-in for tests and dry runs; not a model. |

Credentials are read from the usual environment variables (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, or an `ant auth login` profile).
Extra constructor arguments can be passed from the CLI with
`--backend-arg key=value` (values are parsed as JSON), e.g.
`--backend-arg image_detail=high` or `--backend-arg effort=high`.

Wrap any backend with a persistent cache (`--cache .cache/run.sqlite`, or
`CachedLLM` in Python) to make runs resumable and reproducible; Claude models do
not accept a temperature, so caching is the way to freeze outputs.

## Where the STSG comes from

| `step2_mode` | Needs an STSG? | What happens in Step-2 |
|---|---|---|
| `generate` | no | The model writes the partial STSG of the targets from the frames (STSG-free, as MotionEpic). |
| `select` | yes | The full STSG is shown; the model outputs the relevant partial STSG; boxes are snapped to the reference. |
| `match` | yes | Targets are matched to STSG nodes lexically (+ synonyms), no LLM call. Falls back to `select` when nothing matches. |
| `auto` (default) | – | `select` if an STSG is available, otherwise `generate`. |

STSG sources (`--stsg-provider`, or `stsg=` in `VideoOfThought.run`):

* `static:<dir>` — `<video_id>.json` files, e.g. Action Genome graphs converted
  with `scripts/data/convert_action_genome.py`. Use `StaticSTSGProvider(resample=True)`
  when the graph was built at a different frame rate.
* `vlm` — the backend first parses the full STSG of the clip (chunked, ids kept
  consistent across chunks). This is the prompting analogue of MotionEpic's L2 objective.
* `detector[:<hf id>]` — Grounding DINO (`IDEA-Research/grounding-dino-tiny` by
  default) detects the Step-1 targets plus extra vocabulary in every frame,
  Hungarian IoU matching links them into tracks, and geometric predicates
  connect nearest neighbours. Pass `relation_fn=` to `DetectorSTSGParser` to
  replace geometric predicates by a relation classifier.

## Cost

With `n` options, one VoT instance costs `3 + n + 1 + 1` calls when the first
answer is verified (Steps 1–3, n scoring calls, ranking, verification), plus
`n + 2` per re-execution of Step-4. Steps 4 and 5 are self-contained; only
Steps 1–3 (`context="chain"`) and Step 5 carry the frames. Reduce cost with
`ranking="auto"`, fewer `max_frames`, `frame_max_side`, or `image_detail="low"`
(OpenAI). Per-instance token usage is reported in `ReasoningResult.usage` and
summed in the evaluation report.

## Reproducing the zero-shot protocol with an API model

```bash
python scripts/data/prepare_videoqa.py --benchmark nextqa --raw data/nextqa/raw/val.csv --split val --out data/nextqa/val.jsonl
vot eval --benchmark nextqa --data data/nextqa/val.jsonl --video-root data/nextqa/videos \
    --backend openai:gpt-4o --config configs/vot/api.yaml --method vot \
    --workers 8 --cache .cache/nextqa_gpt4o.sqlite --out results/nextqa/gpt4o_vot.jsonl
# baselines with the same backend and frames
vot eval ... --method cot    --out results/nextqa/gpt4o_cot.jsonl
vot eval ... --method direct --out results/nextqa/gpt4o_direct.jsonl
```

Numbers obtained this way are not those of the paper, which reports MotionEpic.
