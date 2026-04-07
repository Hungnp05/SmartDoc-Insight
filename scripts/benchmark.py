import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.models.ollama_client import OllamaClient
from src.layers.vision_processing import VisionProcessingLayer
from src.layers.knowledge_base import KnowledgeBaseLayer
from src.layers.retrieval_reasoning import RetrievalReasoningLayer

SAMPLE_QA_PAIRS = [
    {
        "question": "Doanh thu quy 3 nam 2024 la bao nhieu?",
        "expected_keywords": ["2.340", "quy 3", "ty"],
        "content_type": "table",
    },
    {
        "question": "Tang truong doanh thu quy 3 so voi quy 2 la bao nhieu phan tram?",
        "expected_keywords": ["14", "14.1", "quy 2", "quy 3"],
        "content_type": "table",
    },
    {
        "question": "Mang kinh doanh nao co ty le tang truong cao nhat trong nam 2024?",
        "expected_keywords": ["bao mat", "42", "cao nhat"],
        "content_type": "table",
    },
    {
        "question": "Tong so nhan vien tinh den cuoi nam 2024 la bao nhieu?",
        "expected_keywords": ["2.412", "nhan vien", "2024"],
        "content_type": "table",
    },
    {
        "question": "Chi phi nhan cong chiem bao nhieu phan tram tong chi phi?",
        "expected_keywords": ["51", "nhan cong", "chi phi"],
        "content_type": "table",
    },
    {
        "question": "Cac rui ro chinh duoc neu trong bao cao la gi?",
        "expected_keywords": ["rui ro", "canh tranh", "nhan su"],
        "content_type": "text",
    },
    {
        "question": "Ke hoach doanh thu nam 2025 la bao nhieu?",
        "expected_keywords": ["11.400", "2025", "30"],
        "content_type": "table",
    },
    {
        "question": "Xu huong tang truong doanh thu tu 2021 den 2024 nhu the nao?",
        "expected_keywords": ["tang", "2021", "2024", "tang truong"],
        "content_type": "figure",
    },
]


def normalize(text: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def evaluate_answer(answer: str, expected_keywords: list) -> float:
    answer_norm = normalize(answer)
    hits = sum(1 for kw in expected_keywords if normalize(kw) in answer_norm)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def run_benchmark(doc_path: str, output_file: str = None, clear_db: bool = True):
    print(f"\nSmartDoc-Insight Benchmark")
    print(f"Document: {doc_path}")
    print(f"Questions: {len(SAMPLE_QA_PAIRS)}\n")

    ollama = OllamaClient(config)
    vision = VisionProcessingLayer(config, ollama)
    kb = KnowledgeBaseLayer(config, ollama)
    rag = RetrievalReasoningLayer(config, kb, ollama)

    if clear_db:
        print("Clearing existing KB...")
        kb.clear_all()

    print("Processing document...")
    start = time.time()
    doc = vision.process_document(doc_path)

    region_types = {}
    for r in doc.all_regions():
        region_types[r.region_type] = region_types.get(r.region_type, 0) + 1
    print(f"Regions: {region_types}")

    chunk_count = kb.ingest_document(doc)
    process_time = time.time() - start
    print(f"Done in {process_time:.1f}s — {chunk_count} chunks indexed\n")
    print("-" * 60)

    results = []
    total_score = 0.0
    type_scores = {"table": [], "figure": [], "text": []}

    for i, qa in enumerate(SAMPLE_QA_PAIRS, 1):
        print(f"[{i}/{len(SAMPLE_QA_PAIRS)}] {qa['question']}")
        q_start = time.time()
        response = rag.query(qa["question"])
        q_time = time.time() - q_start

        score = evaluate_answer(response.answer, qa["expected_keywords"])
        total_score += score

        ctype = qa["content_type"]
        if ctype in type_scores:
            type_scores[ctype].append(score)

        status = "PASS" if score >= 0.5 else "FAIL"
        sources_info = [f"{s.content_type}:p{s.page}" for s in response.sources]

        print(f"  {status}  Score: {score:.0%}  Time: {q_time:.1f}s  Sources: {sources_info}")
        print(f"  Answer: {response.answer[:200].replace(chr(10), ' ')}...")
        print()

        results.append({
            "question": qa["question"],
            "answer": response.answer,
            "score": round(score, 3),
            "latency_s": round(q_time, 2),
            "content_type": ctype,
            "sources": sources_info,
        })

    print("=" * 60)
    avg_score = total_score / len(SAMPLE_QA_PAIRS)
    print(f"OVERALL ACCURACY:   {avg_score:.0%}")
    for ctype, scores in type_scores.items():
        if scores:
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
                "by_type": {k: round(sum(v)/len(v), 3) for k, v in type_scores.items() if v},
                "process_time_s": round(process_time, 1),
                "total_questions": len(SAMPLE_QA_PAIRS),
            },
            "results": results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True, help="Path to PDF document")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--no-clear", action="store_true", help="Skip clearing DB before run")
    args = parser.parse_args()
    run_benchmark(args.doc, args.output, clear_db=not args.no_clear)