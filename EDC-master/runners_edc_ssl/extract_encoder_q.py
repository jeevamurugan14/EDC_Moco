"""
extract_encoder_q.py
Recovers the final encoder_q-only checkpoint from the full training-state
checkpoint (moco_latest.pth), after the original final-save step failed
with a permission error. No retraining needed -- this just re-derives the
exact file that torch.save({"encoder_q": model.encoder_q.state_dict()}, ...)
would have produced, from the already-saved full model state.
"""

import torch
import os

RESUME_PATH = "./MoCo_Weights_FixedAug/moco_latest.pth"
OUTPUT_PATH = "./MoCo_Weights_FixedAug/moco_LungCT_BUSI_APTOS_fixed_aug_200ep_200ep.pth"

print(f"Loading full checkpoint from: {RESUME_PATH}")
ckpt = torch.load(RESUME_PATH, map_location="cpu")

print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"Epoch saved at: {ckpt.get('epoch', 'unknown')}")

full_model_state = ckpt["model"]

# The full MoCo wrapper's state_dict keys are prefixed with "encoder_q."
# (e.g. "encoder_q.layer1.0.conv1.weight"). We need just that sub-module,
# with the "encoder_q." prefix stripped, matching what
# model.encoder_q.state_dict() would directly produce.
encoder_q_state = {}
prefix = "encoder_q."
for k, v in full_model_state.items():
    if k.startswith(prefix):
        new_key = k[len(prefix):]
        encoder_q_state[new_key] = v

print(f"Extracted {len(encoder_q_state)} encoder_q parameters "
      f"(out of {len(full_model_state)} total model parameters)")

if len(encoder_q_state) == 0:
    print("\n[WARNING] No keys matched prefix 'encoder_q.' -- "
          "printing first 10 keys of full model state for inspection:")
    for k in list(full_model_state.keys())[:10]:
        print(f"  {k}")
    raise SystemExit(
        "Extraction found 0 matching keys. The prefix assumption is "
        "wrong -- please share the printed keys above so the script "
        "can be corrected."
    )

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save({"encoder_q": encoder_q_state}, OUTPUT_PATH)
print(f"\nSaved recovered encoder_q checkpoint to:\n  {OUTPUT_PATH}")
print('Checkpoint format: {"encoder_q": state_dict}')
print("Compatible with MoCoResNet50Encoder and edc_ssl.py -- no changes needed.")