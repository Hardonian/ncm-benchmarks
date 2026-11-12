"""Local-first summarizer service around Next-Concept Modeling (XSum model).

One MIT-licensed 88M-param generator + 60M tokenizer (654 MB total), loaded once,
serves /summarize fully offline. Generation logic is a faithful copy of upstream
scripts/eval_shard.py @ c702742 (MIT, Hazy Research) + our device patch, reduced to
single-document serving: same penalties, same allowance, same EOS-cut decoding.

Run:
    NCM_HOME=/path/to/ncm-benchmarks venv/bin/python service/summarize_server.py --port 8031
Measured on HX370 CPU (16 cores, fp32): ~5.7 s cold load, ~2.3-2.5 s per summary.
"""
import argparse
import os
import threading
import time

# Offline-first: pin cache locations BEFORE huggingface_hub resolves them, using
# NCM_HOME (where artifacts/ + hf-home/ live). Machine-default HF_HOME may be
# read-only; the gpt2 pre-tokenizer must come from the local pinned cache.
_HOME = os.environ.get("NCM_HOME", "")
if _HOME:
    os.environ.setdefault("HF_HOME", f"{_HOME}/hf-home")
    os.environ.setdefault("HF_HUB_CACHE", f"{_HOME}/hf-home/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{_HOME}/hf-home/hub")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from omegaconf import OmegaConf

CKPT = None  # set at startup


def load(cfg_root: str, ck: str, device: str):
    import sys
    sys.path.insert(0, os.environ["NCM_ROOT"])
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer

    cfg = OmegaConf.load(cfg_root)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    # weights_only=False required: file is upstream's official training artifact
    # (model dict + inference metadata) from the pinned hazyresearch org. Provenance
    # verified via SHA256SUMS; see NOTICE. Same trust decision upstream eval makes.
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location=device, weights_only=False)  # nosec B614
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    g = cfg.generation
    allow = (sd["inference"]["token_unigram"].to(device).double()[None, :] if g.get("frequency_allowance") else None)
    # revision-pinned: exact commit resolved offline in our verified cache (see NOTICE)
    src = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id),
                                        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    P = int(cfg.model.prefix_len)
    return dict(cfg=cfg, tk=tk, net=net, src=src, P=P,
                EOS=tk.pre_tokenizer.eos_token_id, PAD=tk.pre_tokenizer.pad_token_id,
                device=device, temp=float(g.temperature), top_k=int(g.top_k), top_p=float(g.top_p),
                freq=float(g.get("frequency_penalty", 0.0)), mu=float(g.get("frequency_deficit", 0.0)),
                adj=[float(v) for v in str(g.get("adjacency_penalty", "") or "").split(",")] if g.get("adjacency_penalty") else None,
                allow=allow)


def summarize(S, text: str) -> str:
    ids = S["src"](text)["input_ids"]
    x = torch.tensor([(ids[:S["P"]] + [S["PAD"]] * S["P"])[:S["P"]]], device=S["device"])
    with torch.no_grad():
        g = S["net"].generate(None, x, temperature=S["temp"], top_k=S["top_k"], top_p=S["top_p"], cfg_scale=0.0, cfg_null_eos=S["EOS"])
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = S["tk"].vqvae.decoder(zl[-1])
    # pick_tokens: frequency allowance + adjacency penalty (copied from upstream eval_shard)
    B, T, V = logits.shape
    lg = logits.double()
    allow = (S["allow"][:V] * T) if S["allow"] is not None else None
    al = S["adj"][0] if S["adj"] else 0.0
    aw = int(S["adj"][1]) if S["adj"] and len(S["adj"]) > 1 else 1
    ad = S["adj"][2] if S["adj"] and len(S["adj"]) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if S["freq"] > 0:
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= S["freq"] * excess
            if S["mu"] > 0 and allow is not None:
                step += S["mu"] * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    row = out[0].tolist()
    if S["EOS"] in row:
        row = row[:row.index(S["EOS"])]
    return S["tk"].pre_tokenizer.decode([t for t in row if t not in (S["EOS"], S["PAD"])], skip_special_tokens=True)


def main():
    global CKPT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    home = os.environ["NCM_HOME"]
    device = os.environ.get("NCM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    CKPT = load(f"{home}/artifacts/xsum/config.yaml", f"{home}/artifacts/xsum", device)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[ncm-summarizer] loaded in {load_s}s on {device}", flush=True)

    import fastapi
    import uvicorn
    lock = threading.Lock()
    app = fastapi.FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "device": device, "load_s": load_s, "model": "ncm-xsum-88M"}

    @app.post("/summarize")
    def summarize_ep(payload: dict):
        text = payload.get("text", "").strip()
        if not text:
            return fastapi.responses.JSONResponse({"error": "text required"}, status_code=400)
        t = time.perf_counter()
        with lock:  # one model, thread-safe single-doc generation
            s = summarize(CKPT, text)
        return {"summary": s, "latency_ms": int((time.perf_counter() - t) * 1000), "words": len(s.split())}

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
