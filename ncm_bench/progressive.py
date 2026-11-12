"""Progressive Hierarchical Concept Decoder for Next-Concept Modeling.

Decodes the latent representations across all intermediate code levels (0..K-1),
demonstrating the coarse-to-fine concept evolution from high-level semantic gist
to syntactically detailed text.
"""
from __future__ import annotations

import html
from typing import Any

import torch

from ncm_bench.inference import decode_logits_to_texts, prepare_prefix_batch


def progressive_decode(
    model_bundle: dict,
    text: str | list[int],
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    use_penalties: bool = True,
) -> dict[str, Any]:
    """Generates the full concept sequence and decodes each intermediate scale.

    Returns:
        dict containing 'levels': list of dicts with level index, code count, and decoded text.
    """
    S = model_bundle
    x = prepare_prefix_batch(S, [text])
    eos = S["EOS"] if S["EOS"] is not None else 50256

    with torch.inference_mode():
        g = S["net"].generate(
            None,
            x,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=0.0,
            cfg_null_eos=eos,
        )
        _, zl = S["tk"].vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        level_logits = [S["tk"].vqvae.decoder(z_k) for z_k in zl]

    levels_info = []
    level_dims = S["net"].levels
    raw_bundle = S if use_penalties else {**S, "freq": 0.0, "adj": None}

    for k, logits_k in enumerate(level_logits):
        decoded_text = decode_logits_to_texts(logits_k, raw_bundle)[0]
        levels_info.append(
            {
                "level": k,
                "codes_at_level": int(level_dims[k]),
                "total_codes_so_far": int(sum(level_dims[: k + 1])),
                "text": decoded_text,
                "words": len(decoded_text.split()),
            }
        )

    return {
        "num_levels": len(zl),
        "final_summary": levels_info[-1]["text"],
        "levels": levels_info,
    }


def render_progressive_html(result: dict[str, Any], prompt_text: str = "") -> str:
    """Generates a standalone, XSS-safe dark-themed HTML interactive slider visualizer."""
    levels = result["levels"]
    rows = []
    for lvl in levels:
        pct = round((lvl["level"] + 1) / max(len(levels), 1) * 100)
        escaped_text = html.escape(str(lvl.get("text", "")), quote=True)
        body_html = escaped_text if escaped_text else "<i>(initial semantic codes)</i>"
        rows.append(
            f"""
        <div class="level-card" id="card-{int(lvl['level'])}">
            <div class="level-header">
                <span class="badge">Level {int(lvl['level'])}</span>
                <span class="meta">{int(lvl['codes_at_level'])} codes ({int(lvl['total_codes_so_far'])} accumulated)</span>
                <span class="pct">{pct}% fidelity</span>
            </div>
            <div class="level-body">
                <p class="text-output">{body_html}</p>
            </div>
        </div>
        """
        )

    cards_html = "\n".join(rows)
    safe_prompt = html.escape(prompt_text, quote=True) if prompt_text else "N/A"
    max_level = max(len(levels) - 1, 0)

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
  .prompt-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 24px; font-size: 0.9rem; line-height: 1.5; word-break: break-word; }}
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
    {safe_prompt}
  </div>

  <div class="slider-container">
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="font-weight: 600;">Concept Level: <span id="lvl-label" style="color: #38bdf8;">All Levels</span></span>
        <span style="font-size: 0.85rem; color: #94a3b8;">Slide to inspect individual hierarchy steps</span>
    </div>
    <input type="range" id="levelRange" min="-1" max="{max_level}" value="-1" oninput="updateView(this.value)">
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
