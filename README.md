# FairEval Replication: Testing GLM-5.2 as an LLM Judge

Replication of the FairEval positional bias benchmark ([Wang et al., 2023](https://arxiv.org/abs/2305.17926)) using **GLM-5.2** — Zhipu AI's open source 744B MoE model — as the judge.

Most existing evaluations of LLM-as-a-Judge use proprietary models (GPT-4, Claude). This experiment tests whether a state-of-the-art open source model shows the same positional bias patterns.

---

## What is Positional Bias?

When an LLM is asked to judge two answers, it may prefer whichever answer appears first — regardless of quality. This is positional bias. It means your eval pipeline could be making decisions based on order, not correctness.

We detect it by:
1. Asking the judge to evaluate Answer A vs Answer B
2. Flipping the order: Answer B vs Answer A
3. If the verdict changes → positional bias detected

---

## Setup

- **Judge model:** `z-ai/glm-5.2` via OpenRouter
- **Dataset:** 60 custom questions across 3 difficulty levels and 6 categories
- **Each question evaluated twice** (normal order + flipped order)
- **Temperature:** 0 for deterministic outputs

---

## Dataset

| Difficulty | Count | Description |
|---|---|---|
| Easy | 20 | Obvious quality gap between answers |
| Medium | 20 | One answer is better, but not dramatically |
| Hard | 20 | Both answers are strong, subtle differences |

| Category | Count |
|---|---|
| STEM | 20 |
| Coding | 12 |
| Humanities | 12 |
| Reasoning | 11 |
| Writing | 4 |
| Roleplay | 1 |

Correct answer is mixed across positions (~50% A, ~50% B) to avoid dataset-level position bias.

---

## Results

### Overall

| Metric | GPT-4 (benchmark) | GPT-3.5 (benchmark) | GLM-5.2 (ours) |
|---|---|---|---|
| Consistency | ~85–90% | ~65–70% | **85.0%** |
| Accuracy | ~80–85% | ~65–70% | **81.7%** |
| Bias Score | ~10–15% | ~30–35% | **15.0%** |

### By Difficulty

| Difficulty | Consistency | Accuracy | Bias Score |
|---|---|---|---|
| Easy | 20/20 (100%) | 20/20 (100%) | 0.0% |
| Medium | 19/20 (95%) | 19/20 (95%) | 5.0% |
| Hard | 12/20 (60%) | 10/20 (50%) | **40.0%** |

### By Category

| Category | Consistency | Accuracy | Bias Score |
|---|---|---|---|
| Writing | 4/4 (100%) | 4/4 (100%) | 0.0% |
| Roleplay | 1/1 (100%) | 1/1 (100%) | 0.0% |
| Humanities | 11/12 (91.7%) | 11/12 (91.7%) | 8.3% |
| Reasoning | 10/11 (90.9%) | 10/11 (90.9%) | 9.1% |
| Coding | 10/12 (83.3%) | 9/12 (75.0%) | 16.7% |
| STEM | 15/20 (75.0%) | 14/20 (70.0%) | **25.0%** |

---

## Key Findings

**GLM-5.2 matches GPT-4 on general tasks.** On easy and medium questions it performs within GPT-4's range on all three metrics — impressive for an open source model.

**It degrades on hard technical questions.** At 40% bias score and 50% accuracy on hard questions, GLM-5.2 drops to GPT-3.5 territory. When both answers require deep technical reasoning, it defaults to position rather than quality.

**STEM is the weakest category.** 25% bias score — the highest of any category. This is consistent with the original FairEval finding that LLM judges struggle most when answers require domain-specific evaluation.

**The gap is difficulty-dependent, not model-wide.** GLM-5.2 is a viable open source judge for general evaluation pipelines. The limitation is specific: hard, technically demanding comparisons where both answers are genuinely strong.

---

## Benchmark Reference

- Wang et al. (2023) — [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
- Zheng et al. (2023) — [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)

---

## Repo Structure

```
├── questions.py       # dataset (gitignored, not published — see "Bring Your Own Dataset")
├── judge.py           # Evaluation script with positional bias detection
├── results_*.json     # JSON output from experiment runs (gitignored, not published)
└── README.md
```

---

## Bring Your Own Dataset

The 60-question dataset used for the results above is **not published in this
repo** — `questions.py` and `results_*.json` are gitignored. To replicate the
harness, create your own `questions.py` in the project root with a
`QUESTIONS` list in this shape:

```python
QUESTIONS = [
    {
        "question": "What is machine learning?",
        "answer_a": "A detailed, accurate answer...",
        "answer_b": "A vague or weaker answer...",
        "correct": "a",          # "a" or "b" — which answer is actually better
        "difficulty": "easy",    # any label you want to group by
        "category": "STEM",      # any label you want to group by
    },
    # ...more questions
]
```

Each entry needs `question`, `answer_a`, `answer_b`, `correct`, `difficulty`,
and `category` — `judge.py` reads these fields directly. Mix which side
(`a`/`b`) holds the correct answer roughly 50/50 to avoid dataset-level
position bias skewing your results.

---

## Running It Yourself

```bash
pip install openai python-dotenv
```

Create a `.env` file:
```
OPENROUTER_API_KEY=your_key_here
```

Add your own `questions.py` (see "Bring Your Own Dataset" above), then run:
```bash
python judge.py
```

Results are saved to a timestamped `results_<timestamp>.json` file (gitignored) with per-question breakdowns, consistency flags, and category/difficulty aggregations.

---

## Next Steps

- Add more models for direct comparison (Llama 3.1 70B, Mistral, Qwen2.5)
- Test prompt variations (chain-of-thought, structured rubrics) as bias mitigation
- Expand dataset to domain-specific questions (legal, medical, sports)
