# Jev vs Claude Opus: mortgage document classification (PoC)

A small proof of concept that classifies made-up mortgage documents with two models through [OpenRouter](https://openrouter.ai) and compares speed, token usage, cost and accuracy:

- **Jev** (`typesafe/jev-1.14`): TypeSafe's "System One" decision model. It returns a typed choice instead of free-form text.
- **Claude Opus** (`anthropic/claude-opus-5.5`): a general-purpose LLM prompted to reply with a category name.

> **Disclaimer:** This is a simple PoC / MVP test, not a rigorous benchmark. It uses 16 short, invented documents and a single run. The numbers below are indicative only. More testing is needed for solid metrics: real and messy documents, many more samples, repeated runs, parallel load, and ambiguous edge cases.

## Setup

Requires Python 3.9+ and an OpenRouter API key ([openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)). Check that your account can use the models above.

```bash
git clone https://github.com/clamdown2020/jev-vs-opus-doc-classification.git
cd jev-vs-opus-doc-classification

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your key after OPENROUTER_API_KEY=
python compare_classify.py            # all 16 documents
python compare_classify.py --limit 4  # quick smoke test
```

The key is read from `OPENROUTER_API_KEY` (environment or `.env`). `.env` is git-ignored; never commit it.

## How it works

[compare_classify.py](compare_classify.py) holds 16 invented documents (2 each of 8 types: loan application, pay stub, bank statement, tax return, appraisal report, title report, closing disclosure, insurance declaration). For each document it makes one call to each model, one after the other, and records latency, tokens, cost (from OpenRouter's usage field) and whether the label matched the truth. Edit the `DOCS` list or `CATEGORIES` dict to try your own.

- Jev: `POST /api/v1/systemone` with a `choice` question whose `criteria` are the category descriptions.
- Opus: `POST /api/v1/chat/completions` with a prompt listing the categories, `max_tokens=30`.

## Results (one run, 2026-10-26)

| | Accuracy | Avg time/doc | Total time | Input tokens | Output tokens | Total cost |
|---|---|---|---|---|---|---|
| **Jev** | 16/16 | 0.15 s | 2.4 s | 8,986 | 1,436 | $0.00038 |
| **Opus** | 16/16 | 2.12 s | 33.9 s | 5,996 | 114 | $0.02626 |

### Speed and cost saved with Jev

| | Jev | Opus | Multiple | Saved |
|---|---|---|---|---|
| Total time (16 docs) | 2.4 s | 33.9 s | **~14x faster** | 31.5 s (93%) |
| Total cost (16 docs) | $0.00038 | $0.02626 | **~69x cheaper** | $0.0259 (98.6%) |

Extrapolated per 1,000 documents: about $0.024 for Jev vs about $1.64 for Opus. These figures are linear extrapolations from 16 short documents and will change with real document lengths.

### Caveats

- **Accuracy is a tie, but the test is easy.** Both models got 16/16 on short, clean, clearly different documents. This says little about messy scans, overlapping document types or rare categories.
- **Token counts are not like-for-like.** Jev reported more tokens on both input and output than Opus; the two providers count differently (Jev's input likely includes the category descriptions and internal formatting). Compare cost and time rather than raw token counts.
- **Single run over the network.** Latency varies between runs, with provider load and by location. Calls were sequential, not parallel.
- **Different task shapes.** Jev is built for fast typed decisions (classification, routing, scoring). Opus can also summarize, extract fields and explain, which this test does not exercise.
- **Prices may change.** Costs come from OpenRouter's reported usage at the time of the run.

## Files

- `compare_classify.py`: the comparison script
- `requirements.txt`: dependencies (`httpx`, `python-dotenv`)
- `.env.example`: template for your API key
