"""
SmartDoc-Insight Benchmark
──────────────────────────
Ingests a document and asks a fixed set of questions with known answers.

Scoring: each question lists required facts; each fact is a group of acceptable
alternatives. A fact counts when any alternative appears in the answer after
normalization (lowercase, accents removed, digit separators removed, so
"2.340", "2,340" and "2340" all match). Facts must not appear in the question
itself, so restating the question scores nothing.

The default QA set targets test_data/sample_report.pdf.
"""

import sys
import json
import re
import time
import argparse
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DOC = Path(__file__).parent.parent / "test_data" / "sample_report.pdf"
BENCHMARK_COLLECTION = "smartdoc_benchmark"

SAMPLE_QA_PAIRS = [
    {
        "question": "Doanh thu quý 3 năm 2024 là bao nhiêu?",
        "facts": [["2340"]],
        "content_type": "table",
    },
    {
        "question": "Tăng trưởng doanh thu quý 3 so với quý 2 là bao nhiêu phần trăm?",
        "facts": [["14.1", "14,1"]],
        "content_type": "table",
    },
    {
        "question": "Mảng kinh doanh nào có tỷ lệ tăng trưởng cao nhất trong năm 2024?",
        "facts": [["bao mat"], ["42"]],
        "content_type": "table",
    },
    {
        "question": "Tổng số nhân viên tính đến cuối năm 2024 là bao nhiêu?",
        "facts": [["2412"]],
        "content_type": "table",
    },
    {
        "question": "Chi phí nhân công chiếm bao nhiêu phần trăm tổng chi phí?",
        "facts": [["51.4", "51,4"]],
        "content_type": "table",
    },
    {
        "question": "Các rủi ro chính được nêu trong báo cáo là gì?",
        "facts": [["canh tranh"], ["nhan su", "nhan tai"], ["cong nghe"]],
        "content_type": "text",
    },
    {
        "question": "Kế hoạch doanh thu năm 2025 là bao nhiêu?",
        "facts": [["11400"]],
        "content_type": "table",
    },
    {
        "question": "Biểu đồ doanh thu từ 2021 đến 2024 cho thấy xu hướng gì?",
        "facts": [["tang", "increase", "upward"]],
        "content_type": "figure",
    },
    {
        "question": "Theo biểu đồ doanh thu năm 2024, quý nào đạt doanh thu cao nhất?",
        "facts": [["quy 4", "q4"]],
        "content_type": "figure",
    },
]


def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower().replace("đ", "d"))
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Drop thousands/decimal separators between digits: 2.340 / 2,340 -> 2340, 14.1 / 14,1 -> 141.
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)
    return re.sub(r"\s+", " ", text)


def validate_qa_pairs(qa_pairs: list):
    for qa in qa_pairs:
        q = normalize(qa["question"])
        for fact in qa["facts"]:
            for alt in fact:
                if normalize(alt) in q:
                    raise ValueError(f"Fact '{alt}' appears in the question: {qa['question']}")


def evaluate_answer(answer: str, facts: list) -> float:
    answer_norm = normalize(answer)
    hits = sum(1 for alts in facts if any(normalize(a) in answer_norm for a in alts))
    return hits / len(facts) if facts else 0.0


def run_benchmark(doc_path: str, output_file: str = None, clear_db: bool = True):
    from dataclasses import replace
    from src.config import Config
    from src.pipeline import SmartDocPipeline

    validate_qa_pairs(SAMPLE_QA_PAIRS)

    print("\nSmartDoc-Insight Benchmark")
    print(f"Document: {doc_path}")
    print(f"Questions: {len(SAMPLE_QA_PAIRS)}\n")

    # Separate collection so the benchmark never touches the app's knowledge base.
    bench_config = Config()
    bench_config.chroma = replace(Config.chroma, collection_name=BENCHMARK_COLLECTION)
    pipeline = SmartDocPipeline(bench_config)

    if clear_db:
        print("Clearing benchmark collection...")
        pipeline.clear()

    print("Processing document...")
    start = time.time()
    stats = pipeline.ingest(doc_path)
    process_time = time.time() - start
    print(f"Regions: {stats['regions']}")
    print(f"Done in {process_time:.1f}s — {stats['total_chunks']} chunks indexed\n")
    print("-" * 60)

    results = []
    type_scores = {}

    for i, qa in enumerate(SAMPLE_QA_PAIRS, 1):
        print(f"[{i}/{len(SAMPLE_QA_PAIRS)}] {qa['question']}")
        q_start = time.time()
        response = pipeline.ask(qa["question"], source_filter=Path(doc_path).name)
        q_time = time.time() - q_start

        score = evaluate_answer(response.answer, qa["facts"])
        type_scores.setdefault(qa["content_type"], []).append(score)

        status = "PASS" if score >= 1.0 else ("PARTIAL" if score > 0 else "FAIL")
        sources_info = [f"{s.content_type}:p{s.page}" for s in response.sources]

        print(f"  {status}  Score: {score:.0%}  Time: {q_time:.1f}s  Sources: {sources_info}")
        print(f"  Answer: {response.answer[:200].replace(chr(10), ' ')}...")
        print()

        results.append({
            "question": qa["question"],
            "answer": response.answer,
            "score": round(score, 3),
            "latency_s": round(q_time, 2),
            "content_type": qa["content_type"],
            "sources": sources_info,
        })

    print("=" * 60)
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"OVERALL ACCURACY:   {avg_score:.0%}")
    for ctype, scores in type_scores.items():
        avg = sum(scores) / len(scores)
        bar = "#" * int(avg * 20)
        print(f"  {ctype:<10} {avg:.0%}  [{bar:<20}]  ({len(scores)} questions)")
    print(f"Process time:       {process_time:.1f}s")
    print(f"Avg query time:     {sum(r['latency_s'] for r in results)/len(results):.1f}s")
    print("=" * 60)

    if output_file:
        payload = {
            "summary": {
                "overall": round(avg_score, 3),
                "by_type": {k: round(sum(v) / len(v), 3) for k, v in type_scores.items()},
                "process_time_s": round(process_time, 1),
                "total_chunks": stats["total_chunks"],
                "regions": stats["regions"],
                "total_questions": len(SAMPLE_QA_PAIRS),
            },
            "results": results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=str(DEFAULT_DOC), help="Path to PDF document")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--no-clear", action="store_true", help="Keep the benchmark collection from a previous run")
    args = parser.parse_args()
    run_benchmark(args.doc, args.output, clear_db=not args.no_clear)
