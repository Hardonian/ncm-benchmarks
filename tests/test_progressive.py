import pytest
from ncm_bench.progressive import render_progressive_html


def test_render_progressive_html():
    sample_result = {
        "num_levels": 3,
        "final_summary": "Government spends on renewable energy.",
        "levels": [
            {"level": 0, "codes_at_level": 4, "total_codes_so_far": 4, "text": "spending", "words": 1},
            {"level": 1, "codes_at_level": 6, "total_codes_so_far": 10, "text": "Government spends", "words": 2},
            {"level": 2, "codes_at_level": 10, "total_codes_so_far": 20, "text": "Government spends on renewable energy.", "words": 5},
        ]
    }
    html = render_progressive_html(sample_result, prompt_text="Test prompt")
    assert "<!DOCTYPE html>" in html
    assert "Level 0" in html
    assert "Level 2" in html
    assert "Government spends on renewable energy." in html

