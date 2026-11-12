from ncm_bench.progressive import render_progressive_html


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


