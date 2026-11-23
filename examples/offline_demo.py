"""Walk through the five VoT steps without any model or API key.

The rule-based mock backend answers each step from an illustrative STSG of the
paper's running example, so this script shows the exact prompts, the recycled
intermediate outputs and the verification loop.

    python examples/offline_demo.py            # summary
    python examples/offline_demo.py --trace    # every prompt and response
"""

import argparse
import json
import os

from PIL import Image

from vot import STSG, VideoClip, VideoOfThought, VoTConfig
from vot.backends.mock import RuleBasedMockLLM

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--trace", action="store_true")
ap.add_argument("--reject-first", action="store_true", help="make the first verification fail")
args = ap.parse_args()

with open(os.path.join(HERE, "assets", "garbage_truck_stsg.json")) as f:
    stsg = STSG.from_dict(json.load(f))
clip = VideoClip([Image.new("RGB", (640, 360), (90, 90, 90)) for _ in stsg.frames], [f.timestamp for f in stsg.frames])

model = RuleBasedMockLLM(stsg=stsg, answer_key="Clean", reject_first=args.reject_first)
vot = VideoOfThought(model, VoTConfig(step2_mode="select"))
res = vot(
    clip,
    "What is the relationship between the white truck and this neighborhood?",
    ["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"],
    stsg=stsg,
)
print(res.pretty())
if args.trace:
    for s in res.trace:
        print(f"\n===== {s.step} (round {s.round}) =====\n--- prompt ---\n{s.prompt}\n--- response ---\n{s.response}")
