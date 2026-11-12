"""Local-first summarizer and concept inspection service around Next-Concept Modeling (XSum).

One MIT-licensed 88M-param generator + 60M tokenizer (654 MB total), loaded once,
serves:
- POST /summarize: Standard single-shot summarization.
- POST /mbr_summarize: Minimum Bayes Risk consensus decoding (higher quality).
- POST /progressive_summarize: Hierarchical coarse-to-fine concept inspection across all levels (k=0..8).
- GET  /visualize: Interactive web UI slider visualizer for concept hierarchies.
- GET  /health: Service status, load latency, device, and quantization mode.

Runs fully offline with zero telemetry or network calls.
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

# Offline-first: pin cache locations BEFORE huggingface_hub resolves them
_HOME = os.environ.get("NCM_HOME", str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HOME", f"{_HOME}/hf-home")
os.environ.setdefault("HF_HUB_CACHE", f"{_HOME}/hf-home/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{_HOME}/hf-home/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Ensure repo root and ncm are on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
NCM_ROOT = os.environ.get("NCM_ROOT", str(Path(REPO_ROOT) / "ncm"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if NCM_ROOT not in sys.path:
    sys.path.insert(0, NCM_ROOT)

import torch
from omegaconf import OmegaConf

from ncm_bench.mbr import generate_mbr_summary
from ncm_bench.progressive import progressive_decode, render_progressive_html

CKPT = None  # set at startup


def load(cfg_root: str, ck: str, device: str, quantize: str = "none"):
    from models.next_concept import NextConceptTransformer
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from transformers import AutoTokenizer

    cfg = OmegaConf.load(cfg_root)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location=device, weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()

    if quantize == "int8" and device == "cpu":
        print("[ncm-service] Applying torch.ao.quantization.quantize_dynamic (INT8)...", flush=True)
        net = torch.ao.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)

    g = cfg.generation
    allow = (sd["inference"]["token_unigram"].to(device).double()[None, :] if g.get("frequency_allowance") else None)
    src = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id),
                                        revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    P = int(cfg.model.prefix_len)
    return dict(cfg=cfg, tk=tk, net=net, src=src, P=P,
                EOS=tk.pre_tokenizer.eos_token_id, PAD=tk.pre_tokenizer.pad_token_id,
                device=device, temp=float(g.temperature), top_k=int(g.top_k), top_p=float(g.top_p),
                freq=float(g.get("frequency_penalty", 0.0)), mu=float(g.get("frequency_deficit", 0.0)),
                adj=[float(v) for v in str(g.get("adjacency_penalty", "") or "").split(",")] if g.get("adjacency_penalty") else None,
                allow=allow, quantize=quantize)


def summarize(S, text: str | list[int]) -> str:
    if isinstance(text, str):
        ids = S["src"](text)["input_ids"]
    else:
        ids = list(text)
    x = torch.tensor([(ids[:S["P"]] + [S["PAD"]] * S["P"])[:S["P"]]], device=S["device"])
    with torch.no_grad():
        g = S["net"].generate(None, x, temperature=S["temp"], top_k=S["top_k"], top_p=S["top_p"], cfg_scale=0.0, cfg_null_eos=S["EOS"])
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = S["tk"].vqvae.decoder(zl[-1])
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
    return S["tk"].pre_tokenizer.decode([t for t in row if t not in (S["EOS"], S["PAD"])], skip_special_tokens=True).strip()


def main():
    global CKPT
    ap = argparse.ArgumentParser(description="NCM Local Summarization & Concept Server")
    ap.add_argument("--port", type=int, default=8031)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--quantize", choices=["none", "int8"], default="none", help="Apply dynamic INT8 quantization on CPU")
    ap.add_argument("--device", default=None, help="Target device (cpu, cuda)")
    a = ap.parse_args()

    home = os.environ.get("NCM_HOME", REPO_ROOT)
    device = a.device or os.environ.get("NCM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    CKPT = load(f"{home}/artifacts/xsum/config.yaml", f"{home}/artifacts/xsum", device, quantize=a.quantize)
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[ncm-summarizer] loaded in {load_s}s on {device} (quantize={a.quantize})", flush=True)

    import fastapi
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    lock = threading.Lock()
    app = fastapi.FastAPI(
        title="Next-Concept Modeling Local Service",
        description="Local-first coarse-to-fine concept generation & summarization API",
        version="2.0.0"
    )

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "device": device,
            "load_s": load_s,
            "model": "ncm-xsum-88M",
            "quantize": a.quantize,
        }

    @app.post("/summarize")
    def summarize_endpoint(payload: dict):
        text = payload.get("text", "")
        if not text:
            return JSONResponse({"error": "text required"}, status_code=400)
        t = time.perf_counter()
        with lock:
            s = summarize(CKPT, text)
        return {
            "summary": s,
            "latency_ms": int((time.perf_counter() - t) * 1000),
            "words": len(s.split())
        }

    @app.post("/mbr_summarize")
    def mbr_endpoint(payload: dict):
        text = payload.get("text", "")
        if not text:
            return JSONResponse({"error": "text required"}, status_code=400)
        num_candidates = int(payload.get("num_candidates", 4))
        temperature = float(payload.get("temperature", 0.7))
        utility = payload.get("utility", "rougeL")
        
        t = time.perf_counter()
        with lock:
            res = generate_mbr_summary(
                CKPT,
                text,
                num_candidates=num_candidates,
                temperature=temperature,
                utility=utility
            )
        elapsed_ms = int((time.perf_counter() - t) * 1000)
        res["latency_ms"] = elapsed_ms
        return res

    @app.post("/progressive_summarize")
    def progressive_endpoint(payload: dict):
        text = payload.get("text", "")
        if not text:
            return JSONResponse({"error": "text required"}, status_code=400)
        t = time.perf_counter()
        with lock:
            res = progressive_decode(CKPT, text)
        res["latency_ms"] = int((time.perf_counter() - t) * 1000)
        return res

    @app.get("/visualize", response_class=HTMLResponse)
    def visualize_endpoint(text: str = ""):
        sample = text or (
            "The UK government has announced a major investment into renewable energy infrastructure, "
            "aiming to double wind and solar capacity by 2030. Ministers said the move will lower bills "
            "and cut carbon emissions across England and Wales."
        )
        with lock:
            prog = progressive_decode(CKPT, sample)
        return render_progressive_html(prog, prompt_text=sample)

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
