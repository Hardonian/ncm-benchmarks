"""Progressive Hierarchical Concept Decoder for Next-Concept Modeling.

Decodes the latent representations across all intermediate code levels (0..K-1),
demonstrating the coarse-to-fine concept evolution from high-level semantic gist
to syntactically detailed text.
"""
from typing import Any
import torch


def decode_logits_to_text(logits: torch.Tensor, model_bundle: dict) -> str:
    """Decodes raw logits tensor [1, T, V] into clean text with EOS cut."""
    S = model_bundle
    B, T, V = logits.shape
    lg = logits.double()
    allow = (S["allow"][:V] * T) if S["allow"] is not None else None
    al = S["adj"][0] if S["adj"] else 0.0
    aw = int(S["adj"][1]) if S["adj"] and len(S["adj"]) > 1 else 1
    ad = S["adj"][2] if S["adj"] and len(S["adj"]) > 2 else 1.0

    cnt = torch.zeros(1, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(1, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[0:1, t, :].clone()
        if S["freq"] > 0:
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= S["freq"] * excess
            if S["mu"] > 0 and allow is not None:
                step += S["mu"] * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((1, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(1, 1, dtype=torch.float64, device=lg.device))

    row = out[0].tolist()
    if S["EOS"] in row:
        row = row[:row.index(S["EOS"])]
    clean_ids = [t for t in row if t not in (S["EOS"], S["PAD"])]
    return S["tk"].pre_tokenizer.decode(clean_ids, skip_special_tokens=True).strip()


def progressive_decode(
    model_bundle: dict,
    text: str,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    use_penalties: bool = True
) -> dict[str, Any]:
    """Generates the full concept sequence and decodes each intermediate scale.
    
    Returns:
        dict containing 'levels': list of dicts with level index, code count, and decoded text.
    """
    S = model_bundle
    ids = S["src"](text)["input_ids"]
    x = torch.tensor([(ids[:S["P"]] + [S["PAD"]] * S["P"])[:S["P"]]], device=S["device"])

    with torch.no_grad():
        g = S["net"].generate(
            None,
            x,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=0.0,
            cfg_null_eos=S["EOS"]
        )
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)

    levels_info = []
    level_dims = S["net"].levels

    for k, z_k in enumerate(zl):
        with torch.no_grad():
            logits_k = S["tk"].vqvae.decoder(z_k)
        
        if use_penalties:
            decoded_text = decode_logits_to_text(logits_k, S)
        else:
            raw_ids = logits_k[0].argmax(-1).tolist()
            if S["EOS"] in raw_ids:
                raw_ids = raw_ids[:raw_ids.index(S["EOS"])]
            decoded_text = S["tk"].pre_tokenizer.decode(
                [t for t in raw_ids if t not in (S["EOS"], S["PAD"])],
                skip_special_tokens=True
            ).strip()

        levels_info.append({
            "level": k,
            "codes_at_level": level_dims[k],
            "total_codes_so_far": sum(level_dims[:k+1]),
            "text": decoded_text,
            "words": len(decoded_text.split()),
        })

    return {
        "num_levels": len(zl),
        "final_summary": levels_info[-1]["text"],
        "levels": levels_info,
    }


def render_progressive_html(result: dict[str, Any], prompt_text: str = "") -> str:
    """Generates a standalone, dark-themed HTML interactive slider visualizer."""
    levels = result["levels"]
    rows = []
    for lvl in levels:
        pct = round((lvl["level"] + 1) / len(levels) * 100)
        rows.append(f"""
        <div class="level-card" id="card-{lvl['level']}">
            <div class="level-header">
                <span class="badge">Level {lvl['level']}</span>
                <span class="meta">{lvl['codes_at_level']} codes ({lvl['total_codes_so_far']} accumulated)</span>
                <span class="pct">{pct}% fidelity</span>
            </div>
            <div class="level-body">
                <p class="text-output">{lvl['text'] or '<i>(initial semantic codes)</i>'}</p>
            </div>
        </div>
        """)

    cards_html = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NCM Progressive Coarse-to-Fine Concept Visualization</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; color: #38bdf8; margin-bottom: 8px; }}
  .subtitle {{ color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; }}
  .prompt-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 24px; font-size: 0.9rem; line-height: 1.5; }}
  .slider-container {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 24px; }}
  input[type=range] {{ width: 100%; accent-color: #38bdf8; cursor: pointer; }}
  .level-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 20px; margin-bottom: 12px; transition: all 0.2s ease; }}
  .level-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .badge {{ background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; }}
  .meta {{ color: #64748b; font-size: 0.85rem; }}
  .pct {{ color: #38bdf8; font-weight: 600; font-size: 0.85rem; }}
  .text-output {{ font-size: 1.05rem; line-height: 1.5; color: #f1f5f9; }}
</style>
</head>
<body>
<div class="container">
  <h1>Next-Concept Modeling: Coarse-to-Fine Concept Progression</h1>
  <div class="subtitle">Generating text in discrete concept hierarchies (arXiv:2610.14162). Real hardware reproduction.</div>
  
  <div class="prompt-box">
    <strong style="color: #cbd5e1;">Input Document:</strong><br/>
    {prompt_text or 'N/A'}
  </div>

  <div class="slider-container">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="font-weight: 600;">Concept Level: <span id="lvl-label" style="color: #38bdf8;">All Levels</span></span>
        <span style="font-size: 0.85rem; color: #94a3b8;">Slide to inspect individual hierarchy steps</span>
    </div>
    <input type="range" id="levelRange" min="-1" max="{len(levels)-1}" value="-1" oninput="updateView(this.value)">
  </div>

  <div id="cards-wrapper">
    {cards_html}
  </div>
</div>
<script>
function updateView(val) {{
    const label = document.getElementById('lvl-label');
    const cards = document.querySelectorAll('.level-card');
    if (val == -1) {{
        label.innerText = 'All Levels';
        cards.forEach(c => c.style.display = 'block');
    }} else {{
        label.innerText = 'Level ' + val;
        cards.forEach((c, idx) => {{
            c.style.display = (idx == val) ? 'block' : 'none';
        }});
    }}
}}
</script>
</body>
</html>
"""

