# Service smoke test — 2026-11-11, HX370 CPU (measured, not claimed)

Run:
    NCM_HOME=/home/scott/ai-lab/ncm NCM_ROOT=$NCM_HOME/ncm NCM_DEVICE=cpu \
      venv/bin/python service/summarize_server.py --port 8031 --host 127.0.0.1

## GET /health
    {"ok":true,"device":"cpu","load_s":2.2,"model":"ncm-xsum-88M"}

Warm load 2.2 s (tokenizer + weights already in page cache). The 5.7 s in the
README is the cold-load measurement from the batched timing runs. Both numbers
are ours; cite cold for fresh boots.

## POST /summarize (XSum test doc 0, 428 words, official hazyresearch pickle decoded to raw text)
HTTP 200, end-to-end 2.313 s wall (curl), server-reported latency_ms=2306.

    model:    "The number of beds made available for homeless people living in
               accommodation in Wales has risen in last year, a Welsh housing
               homelessness charity has said."   (26 words)
    gold:     "There is a \"chronic\" need for more housing for prison leavers
               in Wales, according to a charity."

Honest read: fluent, on-topic, correct entity domain, wrong specific subject —
this is exactly the failure mode of an 88M model at R-L ~27. The batch number
(27.10 over 96 docs) is the statistical claim; a single sample is anecdote.
Do not market this as production summarization. Market it as: private, offline,
654 MB, 2.3 s, free forever, good-enough triage summaries.

## Verification properties
- Network: HF_HUB_OFFLINE=1 forced by script (cache vars pinned before hub
  resolves them); the only listening socket was 127.0.0.1:8031.
- Tokenizer revision pinned: gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e.
- bandit: no issues; pip-audit on requirements.txt: no known vulns.
