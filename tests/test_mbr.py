import pytest
from ncm_bench.mbr import select_mbr_centroid, compute_pairwise_rouge_l, compute_pairwise_jaccard


def test_select_mbr_centroid_single():
    cand, idx, scores = select_mbr_centroid(["Only one candidate."])
    assert cand == "Only one candidate."
    assert idx == 0
    assert scores == [1.0]


def test_select_mbr_centroid_consensus():
    # Candidate 1 and 2 are very similar; Candidate 3 is an outlier.
    cands = [
        "The UK government will invest in renewable energy.",
        "The UK government is investing in renewable energy.",
        "A circus elephant danced on a rooftop in Paris.",
    ]
    cand, idx, scores = select_mbr_centroid(cands, utility="rougeL")
    assert idx in (0, 1)  # The consensus candidate should be 0 or 1, not the outlier 2
    assert scores[idx] > scores[2]


def test_select_mbr_jaccard():
    cands = [
        "solar and wind power expansion",
        "solar and wind power investment",
        "totally unrelated recipe for banana bread",
    ]
    cand, idx, scores = select_mbr_centroid(cands, utility="jaccard")
    assert idx in (0, 1)
    assert scores[idx] > scores[2]

