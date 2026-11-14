"""Enterprise coarse-concept triage, early-exit classification, deduplication, and verified extraction."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

from ncm_bench.inference import decode_logits_to_texts, prepare_prefix_batch, summarize_texts
from ncm_bench.longdoc import redact_pii, restore_pii
from ncm_bench.mbr import compute_pairwise_rouge_l

DEFAULT_TRIAGE_LABELS = ["finance", "legal", "security", "operations", "engineering", "general"]

_LABEL_LEXICONS: dict[str, tuple[str, ...]] = {
    "finance": ("revenue", "earnings", "quarter", "margin", "profit", "budget", "invoice", "tax", "bank", "fed"),
    "legal": ("contract", "clause", "liability", "compliance", "indemnify", "lawsuit", "regulation", "gdpr", "hipaa"),
    "security": ("breach", "vulnerability", "cve", "token", "incident", "malware", "auth", "encryption", "audit"),
    "operations": ("supply", "shipment", "outage", "uptime", "sla", "inventory", "logistics", "facility", "incident"),
    "engineering": ("model", "latency", "throughput", "gpu", "cpu", "api", "kubernetes", "deploy", "compiler"),
    "general": ("update", "report", "announcement", "summary", "review", "meeting"),
}

_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "money": re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|m|bn|k))?\b", re.IGNORECASE),
    "dates": re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b"),
    "emails": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "urls": re.compile(r"https?://[^\s<>\"')]+"),
}


def _extract_entities(text: str) -> dict[str, list[str]]:
    extracted: dict[str, list[str]] = {}
    for key, pattern in _ENTITY_PATTERNS.items():
        seen: list[str] = []
        for match in pattern.findall(text):
            val = match.strip()
            if val and val not in seen:
                seen.append(val)
        extracted[key] = seen[:10]
    return extracted


def _normalize_histogram(level_codes: list[list[int]], vocab_size: int = 4096) -> dict[int, float]:
    """Builds a level-weighted L2-normalized sparse frequency vector from hierarchical discrete codes."""
    counts: dict[int, float] = {}
    for lvl_idx, codes in enumerate(level_codes):
        weight = 1.0 / (lvl_idx + 1)
        offset = lvl_idx * vocab_size
        for code in codes:
            bucket = offset + (int(code) % vocab_size)
            counts[bucket] = counts.get(bucket, 0.0) + weight
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm <= 0.0:
        return counts
    return {k: v / norm for k, v in counts.items()}


def _sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(val * b.get(k, 0.0) for k, val in a.items())


def generate_coarse_codes(
    bundle: dict[str, Any],
    texts: list[str | list[int]],
    *,
    max_level: int = 2,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    decode_preview: bool = False,
) -> list[dict[str, Any]]:
    """Runs early-exit NCM generation up to level `max_level`, skipping fine refinement passes."""
    if not texts:
        return []

    net = bundle["net"]
    tk = bundle["tk"]
    levels = list(getattr(net, "levels", (2, 4, 8)))
    clamped_max = max(0, min(int(max_level), len(levels) - 1))

    x = prepare_prefix_batch(bundle, texts)
    eos = bundle.get("EOS")
    eos_id = eos if eos is not None else 50256

    per_level_batches: list[torch.Tensor] = []
    preview_texts: list[str] | None = None

    with torch.inference_mode():
        # Fast direct early-exit path on upstream NextConceptTransformer (skips levels > clamped_max)
        if hasattr(net, "_encode_context") and hasattr(net, "vqvae_local") and clamped_max < len(levels) - 1:
            vq = net.vqvae_local[0].vq
            prefix_len = net.prefix_len
            if getattr(net, "use_ctx_encoder", False):
                prefix_emb = net._encode_context(x[:, :prefix_len])
            else:
                if bool(getattr(net.config, "raw_token_prefix", False)):
                    _enc = net.vqvae_local[0].encoder
                    if (
                        bool(getattr(net.config, "contextual_prefix", False))
                        and len(getattr(_enc, "blocks", [])) > 0
                    ):
                        z_prefix = _enc.forward_nopos(x)[:, :prefix_len, :]
                    else:
                        z_prefix = _enc.wte(x)[:, :prefix_len, :]
                else:
                    z_prefix = net.vqvae_local[0].encode_z(x)[:, :prefix_len, :]
                prefix_emb = net.transformer.wte(z_prefix)
            if not net.use_rope:
                prefix_emb = prefix_emb + net.wpe(net.wp_prefix)

            batch_sz = prefix_emb.shape[0]
            dev = prefix_emb.device
            z_hat = torch.zeros(batch_sz, levels[-1], net.vqvae_local[0].embed_dim, device=dev)

            _pbias = None
            if getattr(net, "prefix_pad_id", None) is not None:
                _pm = x[:, :prefix_len] == net.prefix_pad_id
                _pbias = torch.where(
                    _pm[:, None, None, :],
                    torch.tensor(float("-inf"), device=dev),
                    torch.tensor(0.0, device=dev),
                )
            else:
                _pm = None

            if getattr(net.config, "use_level_encoding", False):
                level_pos_embed = net.transformer.wle(net.lvl_1L)
            else:
                level_pos_embed = torch.zeros(1, net.L - levels[0], net.config.n_embd, device=dev)
            if not net.use_rope:
                level_pos_embed = level_pos_embed + net.wpe(net.wp_1L)

            # Level 0 forward pass
            if getattr(net, "query_level0", False):
                l0 = levels[0]
                _q0 = net.build_l0_query_input(batch_sz, prefix_emb.dtype)
                if getattr(net, "use_post_upsample_conv", False):
                    _q0 = net._apply_post_upsample_conv_single(_q0, 0)
                q = net.transformer.wte(_q0)
                if not net.use_rope:
                    q = q + net.wpe(net.wp_1L[:, :l0])
                if net.config.use_level_encoding:
                    q = q + net.transformer.wle(net.lvl_1L[:, :l0])
                h = torch.cat([prefix_emb, q], dim=1)
                m0 = torch.zeros(1, 1, prefix_len + l0, prefix_len + l0, device=dev)
                m0[:, :, :prefix_len, prefix_len:] = float("-inf")
                if _pbias is not None:
                    m0 = m0 + F.pad(_pbias, (0, l0))
                rope_0 = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
                for block in net.transformer.blocks:
                    h = block(h, attn_mask=m0, rope_cos_sin=rope_0)
                h = net.transformer.ln_f(h)
                idx_cur = net._sample_level(net.lm_heads[0](h[:, prefix_len:, :]), temperature, top_k, top_p)
            else:
                rope_p = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
                h = prefix_emb
                pfx_mask = torch.zeros(1, 1, prefix_len, prefix_len, device=dev)
                if _pbias is not None:
                    pfx_mask = pfx_mask + _pbias
                for block in net.transformer.blocks:
                    h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
                h = net.transformer.ln_f(h)
                l0_logits = net.lm_head_l0(net._level0_slice(h, _pm if net.prefix_pad_id is not None else None))
                idx_cur = net._sample_level(l0_logits, temperature, top_k, top_p)

            per_level_batches.append(idx_cur)
            level_acum = levels[0] if getattr(net, "query_level0", False) else 0
            for level_idx in range(clamped_max):
                z_q_level = vq.embeddings[level_idx](idx_cur)
                z_hat, lvl_x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
                if getattr(net, "use_post_upsample_conv", False):
                    lvl_x = net._apply_post_upsample_conv_single(lvl_x, level_idx + (1 - net._lvl_first))
                lvl_x = net.transformer.wte(lvl_x) + level_pos_embed[:, level_acum : level_acum + levels[level_idx + 1], :]
                hidx = level_idx + (1 - net._lvl_first)
                logits = net._decode_level_logits(lvl_x, prefix_emb, level_idx, _pbias, hidx=hidx)
                idx_cur = net._sample_level(logits, temperature, top_k, top_p)
                per_level_batches.append(idx_cur)
                level_acum += levels[level_idx + 1]

            if decode_preview and hasattr(tk, "vqvae"):
                # Decode final truncated latent z_hat accumulated up to clamped_max + 1
                z_q_last = vq.embeddings[clamped_max](idx_cur)
                if clamped_max < len(levels) - 1:
                    z_hat, _ = vq.get_next_autoregressive_input(clamped_max, z_hat, z_q_last)
                logits_prev = tk.vqvae.decoder(z_hat)
                preview_texts = decode_logits_to_texts(logits_prev, bundle)
        else:
            full_codes = net.generate(
                None,
                x,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=0.0,
                cfg_null_eos=eos_id,
            )
            offset = 0
            for k_idx in range(clamped_max + 1):
                width = levels[k_idx]
                per_level_batches.append(full_codes[:, offset : offset + width])
                offset += width
            if decode_preview and hasattr(tk, "vqvae"):
                _, z_levels = tk.vqvae.vq.idx_to_zhats(full_codes, return_zlevels=True)
                preview_texts = decode_logits_to_texts(tk.vqvae.decoder(z_levels[clamped_max]), bundle)

    out: list[dict[str, Any]] = []
    batch_len = per_level_batches[0].shape[0]
    for b_idx in range(batch_len):
        code_lists = [[int(v) for v in lvl_t[b_idx].tolist()] for lvl_t in per_level_batches]
        flat_codes = [c for sub in code_lists for c in sub]
        concept_hash = hashlib.sha256(",".join(map(str, flat_codes)).encode("utf-8")).hexdigest()[:16]
        entry: dict[str, Any] = {
            "max_level": clamped_max,
            "total_levels": len(levels),
            "passes_executed": clamped_max + 1,
            "levels": code_lists,
            "concept_hash": concept_hash,
            "histogram": _normalize_histogram(code_lists),
        }
        if preview_texts is not None:
            entry["preview_text"] = preview_texts[b_idx]
        out.append(entry)
    return out


class ConceptIndex:
    """Zero-embedding discrete concept index for semantic deduplication and nearest-neighbor lookup."""

    def __init__(self) -> None:
        self._items: list[tuple[str, dict[int, float], dict[str, Any]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, doc_id: str, signature: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        hist = signature["histogram"]
        meta = dict(metadata or {})
        meta.setdefault("concept_hash", signature.get("concept_hash", ""))
        self._items.append((doc_id, hist, meta))

    def query(
        self,
        signature: dict[str, Any],
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict[str, Any]]:
        target = signature["histogram"]
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for doc_id, hist, meta in self._items:
            sim = _sparse_cosine(target, hist)
            if sim >= min_similarity:
                scored.append((sim, doc_id, meta))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {"doc_id": doc_id, "similarity": round(sim, 4), "metadata": meta}
            for sim, doc_id, meta in scored[:top_k]
        ]

    def is_duplicate(self, signature: dict[str, Any], *, threshold: float = 0.90) -> bool:
        hits = self.query(signature, top_k=1, min_similarity=threshold)
        return len(hits) > 0


class ConceptClassifier:
    """Prototype centroid classifier operating purely on coarse Level 0..k discrete concept codes."""

    def __init__(self, escalation_confidence: float = 0.45) -> None:
        self.escalation_confidence = escalation_confidence
        self.centroids: dict[str, dict[int, float]] = {}

    def fit(self, signatures: list[dict[str, Any]], labels: list[str]) -> "ConceptClassifier":
        if len(signatures) != len(labels):
            raise ValueError("signatures and labels must match in length")
        if not signatures:
            raise ValueError("signatures must not be empty")

        grouped: dict[str, list[dict[int, float]]] = {}
        for sig, lbl in zip(signatures, labels, strict=True):
            grouped.setdefault(lbl, []).append(sig["histogram"])

        self.centroids = {}
        for lbl, hists in grouped.items():
            summed: dict[int, float] = {}
            for h in hists:
                for k, v in h.items():
                    summed[k] = summed.get(k, 0.0) + v
            norm = math.sqrt(sum(v * v for v in summed.values())) or 1.0
            self.centroids[lbl] = {k: v / norm for k, v in summed.items()}
        return self

    def predict_one(self, signature: dict[str, Any], *, text_hint: str = "") -> dict[str, Any]:
        if not self.centroids:
            return _heuristic_route(text_hint, escalation_threshold=self.escalation_confidence)

        scores = {lbl: _sparse_cosine(signature["histogram"], c) for lbl, c in self.centroids.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = max(0.0, best_score - second_score)
        # Softmax over temperature-scaled cosine similarities for calibrated confidence
        temp = 0.25
        exp_scores = {lbl: math.exp(sc / temp) for lbl, sc in ranked}
        z = sum(exp_scores.values()) or 1.0
        confidence = exp_scores[best_label] / z

        return {
            "label": best_label,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "escalate": confidence < self.escalation_confidence,
            "scores": {lbl: round(sc, 4) for lbl, sc in ranked},
            "concept_hash": signature.get("concept_hash", ""),
            "early_exit_passes": signature.get("passes_executed", 0),
        }


def _heuristic_route(text: str, *, labels: list[str] | None = None, escalation_threshold: float = 0.45) -> dict[str, Any]:
    target_labels = labels or DEFAULT_TRIAGE_LABELS
    lower = text.lower()
    tokens = re.findall(r"[a-z0-9]+", lower)
    counts = Counter(tokens)
    raw_scores: dict[str, float] = {}
    for lbl in target_labels:
        lex = _LABEL_LEXICONS.get(lbl.lower(), (lbl.lower(),))
        score = float(sum(counts.get(w, 0) for w in lex)) + (1.5 if lbl.lower() in lower else 0.0)
        raw_scores[lbl] = score

    ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
    best_lbl, best_raw = ranked[0]
    second_raw = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(raw_scores.values())
    confidence = (best_raw / total) if total > 0 else (1.0 / max(len(target_labels), 1))
    margin = (best_raw - second_raw) / max(total, 1.0)
    return {
        "label": best_lbl if total > 0 else target_labels[0],
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "escalate": confidence < escalation_threshold or total == 0,
    }


def validate_and_repair_triage_record(
    record: dict[str, Any],
    *,
    source_text: str,
    allowed_labels: list[str] | None = None,
) -> dict[str, Any]:
    """Deterministically validates and repairs structured triage fields against schema & source constraints."""
    labels = allowed_labels or DEFAULT_TRIAGE_LABELS
    repaired = dict(record)
    checks_passed = 0
    total_checks = 4

    # 1) Label must belong to allowed taxonomy
    cat = str(repaired.get("category", "")).strip()
    if cat not in labels:
        cat = _heuristic_route(source_text, labels=labels)["label"]
        repaired["category"] = cat
    if cat in labels:
        checks_passed += 1

    # 2) Priority constrained to valid tier
    prio = str(repaired.get("priority", "")).lower().strip()
    if prio not in {"low", "medium", "high", "critical"}:
        lower = source_text.lower()
        if any(w in lower for w in ("breach", "outage", "critical", "urgent", "immediate", "lawsuit")):
            prio = "high"
        elif any(w in lower for w in ("audit", "deadline", "quarter", "incident")):
            prio = "medium"
        else:
            prio = "low"
        repaired["priority"] = prio
    if prio in {"low", "medium", "high", "critical"}:
        checks_passed += 1

    # 3) Grounded non-empty summary
    summary = " ".join(str(repaired.get("summary", "")).split())
    if not summary:
        first_sentence = re.split(r"(?<=[.!?])\s+", source_text.strip())[0]
        summary = first_sentence[:240]
        repaired["summary"] = summary
    if summary:
        checks_passed += 1

    # 4) Deterministic entity grounding vs source text
    source_entities = _extract_entities(source_text)
    entities = repaired.get("entities")
    if not isinstance(entities, dict):
        repaired["entities"] = source_entities
    else:
        for k, vals in source_entities.items():
            existing = entities.get(k, [])
            if not isinstance(existing, list) or not existing:
                entities[k] = vals
    checks_passed += 1

    repaired["schema_valid"] = checks_passed == total_checks
    repaired["schema_score"] = round(checks_passed / total_checks, 4)
    return repaired


def extract_structured_triage(
    bundle: dict[str, Any],
    text: str,
    *,
    labels: list[str] | None = None,
    classifier: ConceptClassifier | None = None,
    max_level: int = 2,
    best_of_k: int = 3,
    temperature: float = 0.5,
    redact: bool = False,
    restore: bool = False,
) -> dict[str, Any]:
    """Runs early-exit coarse classification + parallel Best-of-K verified structured triage extraction."""
    target_labels = labels or DEFAULT_TRIAGE_LABELS
    proc_text = text
    pii_map: dict[str, str] = {}
    if redact:
        proc_text, pii_map = redact_pii(text)

    # 1) Early-exit coarse concept extraction (Levels 0..max_level)
    coarse_sig = generate_coarse_codes(bundle, [proc_text], max_level=max_level)[0]
    if classifier is not None and classifier.centroids:
        route = classifier.predict_one(coarse_sig, text_hint=proc_text)
    else:
        route = _heuristic_route(proc_text, labels=target_labels)

    # 2) Parallel Best-of-K candidate summaries scored by self-consistency + source entity overlap
    if best_of_k <= 1:
        candidates = summarize_texts(bundle, [proc_text], temperature=0.0)
    else:
        candidates = [summarize_texts(bundle, [proc_text], temperature=0.0)[0]]
        if best_of_k > 1:
            candidates.extend(
                summarize_texts(bundle, [proc_text] * (best_of_k - 1), temperature=temperature)
            )

    best_summary = candidates[0]
    if len(candidates) > 1:
        peer_totals = compute_pairwise_rouge_l(candidates)
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        best_score = -1.0
        for idx_c, cand in enumerate(candidates):
            peer_sim = peer_totals[idx_c] / max(len(candidates) - 1, 1)
            grounding = scorer.score(proc_text[:600], cand)["rougeL"].fmeasure
            score = 0.7 * peer_sim + 0.3 * grounding
            if score > best_score:
                best_score = score
                best_summary = cand

    if restore and pii_map:
        best_summary = restore_pii(best_summary, pii_map)

    raw_record = {
        "category": route["label"],
        "confidence": route["confidence"],
        "escalate_to_frontier": route["escalate"],
        "summary": best_summary,
        "entities": _extract_entities(proc_text if not restore else text),
        "concept_hash": coarse_sig["concept_hash"],
        "early_exit_passes": coarse_sig["passes_executed"],
        "total_concept_levels": coarse_sig["total_levels"],
        "candidates_evaluated": len(candidates),
        "pii_redactions": len(pii_map),
    }
    return validate_and_repair_triage_record(
        raw_record,
        source_text=proc_text if not restore else text,
        allowed_labels=target_labels,
    )
