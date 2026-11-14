from types import SimpleNamespace

import torch

from ncm_bench.progressive import progressive_decode, render_progressive_html


def test_progressive_decode_levels_with_fake_bundle():
    class FakePreTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, _text):
            return [4, 5, 6]

        def decode(self, ids, skip_special_tokens=True):
            vocab = {4: "Budget", 5: "report", 6: "approved"}
            return " ".join(vocab.get(i, "") for i in ids if not skip_special_tokens or i not in (0, 2)).strip()

    class FakeVQ:
        def idx_to_zhats(self, g, return_zlevels=True):
            assert return_zlevels
            b = g.shape[0]
            return g, [torch.full((b, 3), k + 1) for k in range(3)]

    class FakeVQVAE:
        vq = FakeVQ()

        def decoder(self, z_k):
            b, seq_len = z_k.shape
            lvl = int(z_k[0, 0].item())
            logits = torch.full((b, seq_len, 10), -10.0)
            tokens_by_level = {
                1: [4, 2, 0],
                2: [4, 5, 2],
                3: [4, 5, 6],
            }[lvl]
            for pos, tok in enumerate(tokens_by_level):
                logits[0, pos, tok] = 10.0
            return logits

    class FakeNet:
        levels = [2, 4, 8]

        def generate(self, _ctx, x, **_kwargs):
            return torch.zeros((x.shape[0], 14), dtype=torch.long)

    bundle = {
        "P": 4,
        "EOS": 2,
        "PAD": 0,
        "src": None,
        "tk": SimpleNamespace(pre_tokenizer=FakePreTokenizer(), vqvae=FakeVQVAE()),
        "net": FakeNet(),
        "device": "cpu",
        "freq": 0.0,
        "adj": None,
        "cut_at_eos": True,
    }
    res = progressive_decode(bundle, "Budget report", use_penalties=False)
    assert res["num_levels"] == 3
    assert [lvl["codes_at_level"] for lvl in res["levels"]] == [2, 4, 8]
    assert [lvl["total_codes_so_far"] for lvl in res["levels"]] == [2, 6, 14]
    assert res["final_summary"] == "Budget report approved"


def test_render_progressive_html():
    sample_result = {
        "num_levels": 3,
        "final_summary": "Government spends on renewable energy.",
        "levels": [
            {"level": 0, "codes_at_level": 4, "total_codes_so_far": 4, "text": "spending", "words": 1},
            {"level": 1, "codes_at_level": 6, "total_codes_so_far": 10, "text": "Government spends", "words": 2},
            {"level": 2, "codes_at_level": 10, "total_codes_so_far": 20, "text": "Government spends on renewable energy.", "words": 5},
        ],
    }
    html_doc = render_progressive_html(sample_result, prompt_text="Test prompt")
    assert "<!DOCTYPE html>" in html_doc
    assert "Level 0" in html_doc
    assert "Level 2" in html_doc
    assert "Government spends on renewable energy." in html_doc


def test_render_progressive_html_escapes_xss():
    malicious = '<script>alert("xss")</script>'
    sample_result = {
        "num_levels": 1,
        "final_summary": malicious,
        "levels": [
            {"level": 0, "codes_at_level": 4, "total_codes_so_far": 4, "text": malicious, "words": 1},
        ],
    }
    html_doc = render_progressive_html(sample_result, prompt_text=malicious)
    assert '<script>alert("xss")</script>' not in html_doc
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html_doc


