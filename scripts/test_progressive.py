import os
import sys

import torch

os.environ["NCM_HOME"] = os.getcwd()
os.environ["NCM_ROOT"] = os.path.join(os.getcwd(), "ncm")
os.environ["NCM_DEVICE"] = "cpu"
sys.path.insert(0, os.getcwd())

from service.summarize_server import load  # noqa: E402

S = load("artifacts/xsum/config.yaml", "artifacts/xsum", "cpu")
text = (
    "The UK government has announced a major investment into renewable energy infrastructure, "
    "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
    "and cut emissions across England and Wales."
)

ids = S["src"](text)["input_ids"]
x = torch.tensor([(ids[:S["P"]] + [S["PAD"]] * S["P"])[:S["P"]]], device="cpu")

with torch.no_grad():
    g = S["net"].generate(None, x, temperature=0.0, cfg_scale=0.0, cfg_null_eos=S["EOS"])
    _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    print(f"Number of progressive zlevels: {len(zl)}")
    for k, z_k in enumerate(zl):
        logits_k = S["tk"].vqvae.decoder(z_k)
        pred_ids = logits_k[0].argmax(-1).tolist()
        if S["EOS"] in pred_ids:
            pred_ids = pred_ids[:pred_ids.index(S["EOS"])]
        decoded = S["tk"].pre_tokenizer.decode(
            [t for t in pred_ids if t not in (S["EOS"], S["PAD"])],
            skip_special_tokens=True
        ).strip()
        n_codes = S["net"].levels[k]
        print(f"Level {k:2d} (codes={n_codes:2d}): \"{decoded}\"")
