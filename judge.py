import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from questions import QUESTIONS  # questions.py is gitignored (private dataset) — supply your own, see README
from collections import defaultdict

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "z-ai/glm-5.2"

def judge(question, answer_a, answer_b):
    prompt = f"""You are an impartial judge evaluating two answers to a question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better? Respond with ONLY 'A' or 'B'. No explanation."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()


def run_experiment():
    results = []
    by_difficulty = defaultdict(list)
    by_category = defaultdict(list)

    for i, q in enumerate(QUESTIONS):
        print(f"Q{i+1} [{q['difficulty'].upper()}][{q['category']}]: {q['question'][:60]}...")

        # Round 1: A then B
        verdict_1 = judge(q["question"], q["answer_a"], q["answer_b"])

        # Round 2: flipped — B then A
        verdict_2_raw = judge(q["question"], q["answer_b"], q["answer_a"])
        verdict_2 = "A" if verdict_2_raw == "B" else "B"

        consistent = verdict_1 == verdict_2
        accurate = verdict_1 == q["correct"].upper() and verdict_2 == q["correct"].upper()

        print(f"  Normal: {verdict_1} | Flipped: {verdict_2} | Consistent: {consistent} | Accurate: {accurate}")

        result = {
            "question_id": i + 1,
            "question": q["question"],
            "answer_a": q["answer_a"],
            "answer_b": q["answer_b"],
            "correct": q["correct"].upper(),
            "difficulty": q["difficulty"],
            "category": q["category"],
            "verdict_normal": verdict_1,
            "verdict_flipped": verdict_2,
            "verdict_flipped_raw": verdict_2_raw,
            "consistent": consistent,
            "accurate": accurate,
            "bias_detected": not consistent
        }

        results.append(result)
        by_difficulty[q["difficulty"]].append(result)
        by_category[q["category"]].append(result)

    # ── OVERALL ──────────────────────────────────────────────
    total = len(results)
    print("\n" + "="*50)
    print(f"OVERALL RESULTS ({total} questions)")
    print("="*50)
    consistent_count = sum(r["consistent"] for r in results)
    accurate_count = sum(r["accurate"] for r in results)
    print(f"Consistency : {consistent_count}/{total} ({100*consistent_count/total:.1f}%)")
    print(f"Accuracy    : {accurate_count}/{total} ({100*accurate_count/total:.1f}%)")
    print(f"Bias Score  : {100*(total-consistent_count)/total:.1f}% of verdicts flipped when order changed")

    # ── BY DIFFICULTY ────────────────────────────────────────
    print("\n" + "-"*50)
    print("BY DIFFICULTY")
    print("-"*50)
    for level in ["easy", "medium", "hard"]:
        r = by_difficulty[level]
        n = len(r)
        c = sum(x["consistent"] for x in r)
        a = sum(x["accurate"] for x in r)
        print(f"{level.upper():8} | Consistency: {c}/{n} ({100*c/n:.1f}%) | Accuracy: {a}/{n} ({100*a/n:.1f}%) | Bias: {100*(n-c)/n:.1f}%")

    # ── BY CATEGORY ──────────────────────────────────────────
    print("\n" + "-"*50)
    print("BY CATEGORY")
    print("-"*50)
    for cat in sorted(by_category.keys()):
        r = by_category[cat]
        n = len(r)
        c = sum(x["consistent"] for x in r)
        a = sum(x["accurate"] for x in r)
        print(f"{cat:12} | Consistency: {c}/{n} ({100*c/n:.1f}%) | Accuracy: {a}/{n} ({100*a/n:.1f}%) | Bias: {100*(n-c)/n:.1f}%")

    print("="*50)

    return results, by_difficulty, by_category


def save_results(results, by_difficulty, by_category):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"

    total = len(results)
    consistent_count = sum(r["consistent"] for r in results)
    accurate_count = sum(r["accurate"] for r in results)

    output = {
        "model": MODEL,
        "timestamp": timestamp,
        "total_questions": total,
        "summary": {
            "consistency_rate": round(consistent_count / total, 4),
            "accuracy_rate": round(accurate_count / total, 4),
            "bias_score": round((total - consistent_count) / total, 4)
        },
        "by_difficulty": {
            level: {
                "total": len(r),
                "consistent": sum(x["consistent"] for x in r),
                "accurate": sum(x["accurate"] for x in r),
                "bias_score": round((len(r) - sum(x["consistent"] for x in r)) / len(r), 4)
            }
            for level, r in by_difficulty.items()
        },
        "by_category": {
            cat: {
                "total": len(r),
                "consistent": sum(x["consistent"] for x in r),
                "accurate": sum(x["accurate"] for x in r),
                "bias_score": round((len(r) - sum(x["consistent"] for x in r)) / len(r), 4)
            }
            for cat, r in by_category.items()
        },
        "failed_consistency": [r for r in results if not r["consistent"]],
        "failed_accuracy": [r for r in results if not r["accurate"]],
        "all_results": results
    }

    # NOTE: this embeds full question/answer text from questions.py — gitignored (results_*.json) so the dataset stays private
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")
    print(f"Questions with bias detected : {len(output['failed_consistency'])}")
    print(f"Questions answered incorrectly: {len(output['failed_accuracy'])}")


if __name__ == "__main__":
    results, by_difficulty, by_category = run_experiment()
    save_results(results, by_difficulty, by_category)