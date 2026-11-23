"""VoT with an API model.

    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY=...
    python examples/quickstart_api.py --video path/to/video.mp4 --backend openai:gpt-4o
    python examples/quickstart_api.py --video v.mp4 --backend anthropic:claude-opus-5-5
    # any OpenAI-compatible server, e.g. vLLM serving Qwen2.5-VL:
    python examples/quickstart_api.py --video v.mp4 --backend openai:Qwen/Qwen2.5-VL-72B-Instruct \
        --base-url http://localhost:8000/v1
    # ground Step-2 on an open-vocabulary detector + tracker instead of model-written boxes:
    python examples/quickstart_api.py --video v.mp4 --backend openai:gpt-4o --stsg-provider detector
"""

import argparse

from vot import VideoOfThought, VoTConfig, build_backend
from vot.grounding import build_provider

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--question", default="What is the relationship between the white truck and this neighborhood?")
ap.add_argument("--options", nargs="*", default=["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"])
ap.add_argument("--backend", default="openai:gpt-4o")
ap.add_argument("--base-url", default=None)
ap.add_argument("--stsg-provider", default=None, help="vlm | detector")
ap.add_argument("--cache", default=".cache/quickstart.sqlite")
args = ap.parse_args()

kw = {"base_url": args.base_url} if args.base_url else {}
model = build_backend(args.backend, cache=args.cache, **kw)
provider = build_provider(args.stsg_provider, model=model) if args.stsg_provider else None
cfg = VoTConfig(fps=1.0, max_frames=16, step2_mode="select" if provider else "generate")

vot = VideoOfThought(model, cfg, stsg_provider=provider)
res = vot(args.video, args.question, args.options or None)
print(res.pretty())
with open("vot_result.json", "w") as f:
    f.write(res.to_json(indent=2))
print("full trace -> vot_result.json")
